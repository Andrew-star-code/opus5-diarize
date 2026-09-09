"""Подключение к SQLite.

БД лежит в именованном docker-томе (ext4 внутри WSL2), а не в bind-mount
на диск Windows: SQLite поверх virtiofs/9p периодически ловит
"disk I/O error" на блокировках.
"""
import logging
from collections.abc import Iterator

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session as DBSession
from sqlmodel import SQLModel, create_engine

from .config import settings

log = logging.getLogger(__name__)

settings.db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{settings.db_path}",
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True,
)


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_connection, connection_record) -> None:
    cur = dbapi_connection.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


def init_db() -> None:
    # Импорт ради регистрации таблиц в метаданных SQLModel.
    from . import models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    """Догоняет схему на уже существующей базе.

    create_all() создаёт недостающие таблицы, но ничего не делает с
    таблицей, которая уже есть: новые поля модели в неё не попадут, и
    приложение упадёт на первом же запросе. Полноценная система миграций
    для одного файла SQLite избыточна, а вот добавить недостающие
    колонки — ровно то, что нужно.

    Удаления и переименования сюда сознательно не входят: они требуют
    решения, что делать с данными, и это не работа для автозапуска.
    """
    from sqlalchemy import inspect

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    for table in SQLModel.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        present = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name not in present:
                _add_column(table.name, column)
            # Даже у существующей колонки в старых строках может лежать
            # NULL — от предыдущего добавления. Модель такого не ждёт и
            # падает на сериализации, поэтому дозаполняем всегда.
            _backfill(table.name, column)


def _add_column(table_name: str, column) -> None:
    """Добавляет одну колонку в отдельной транзакции.

    Отдельной — потому что в общей одна неудачная команда переводит
    транзакцию в сорванное состояние, и при выходе из блока откатывается
    всё, включая уже успевшие пройти изменения. Именно так молча
    потерялось заполнение sessions.prompt: споткнулись на колонке с
    именем order, а откатилось всё разом.
    """
    from sqlalchemy import text

    try:
        ddl_type = column.type.compile(engine.dialect)
        default = _sql_default(column)
        log.info("Добавляю колонку %s.%s", table_name, column.name)
        with engine.begin() as connection:
            connection.execute(
                text(
                    f'ALTER TABLE "{table_name}" '
                    f'ADD COLUMN "{column.name}" {ddl_type}{default}'
                )
            )
    except Exception:
        log.exception("Не удалось добавить колонку %s.%s", table_name, column.name)


def _zero_value(column):
    """Нейтральное значение типа колонки: '' для строк, 0 для чисел.

    Тип определяем по классу, а не через python_type: у AutoString из
    SQLModel — а это все строковые поля модели — python_type бросает
    NotImplementedError. Именно на этом молча провалилось заполнение
    sessions.prompt, тогда как числовые колонки прошли нормально.

    Для дат и прочего возвращаем None — такие поля объявлены Optional,
    и NULL в них законен.
    """
    from sqlalchemy import types as sqltypes

    base = _base_type(column.type)
    if isinstance(base, sqltypes.Boolean):
        return 0
    if isinstance(base, (sqltypes.String, sqltypes.Text)):
        return ""
    if isinstance(base, (sqltypes.Integer, sqltypes.Numeric, sqltypes.Float)):
        return 0
    return None


def _base_type(type_):
    """Снимает обёртки TypeDecorator и возвращает исходный тип SQLAlchemy.

    Строковые поля SQLModel имеют тип AutoString, а он не наследник
    String, а обёртка над ним, поэтому прямая проверка isinstance даёт
    ложное «не знаю» — из-за чего заполнение sessions.prompt молча
    ничего не делало.
    """
    from sqlalchemy import types as sqltypes

    for _ in range(5):  # вложенность обёрток на практике не больше одной
        if not isinstance(type_, sqltypes.TypeDecorator):
            break
        type_ = getattr(type_, "impl_instance", None) or type_.impl
        if isinstance(type_, type):
            type_ = type_()
    return type_


def _backfill(table_name: str, column) -> None:
    """Заменяет NULL в колонке на значение по умолчанию из модели.

    Только там, где модель NULL не допускает. У Optional-полей NULL —
    это значение со смыслом: audio_path без записи, error без ошибки.
    Затирать их пустой строкой было бы порчей данных.

    Имена в кавычках: среди полей встречается order, а это зарезервированное
    слово SQL, и без кавычек запрос просто не разбирается.
    """
    if column.nullable:
        return

    filler = _zero_value(column)
    if filler is None:
        return
    from sqlalchemy import text

    try:
        with engine.begin() as connection:
            result = connection.execute(
                text(
                    f'UPDATE "{table_name}" SET "{column.name}" = :value '
                    f'WHERE "{column.name}" IS NULL'
                ),
                {"value": filler},
            )
        if result.rowcount:
            log.info(
                "Заполнено пустых значений в %s.%s: %d",
                table_name, column.name, result.rowcount,
            )
    except Exception:
        log.exception("Не удалось заполнить %s.%s", table_name, column.name)


def _sql_default(column) -> str:
    """Значение по умолчанию для ALTER TABLE ADD COLUMN.

    SQLite требует, чтобы у добавляемой NOT NULL колонки был DEFAULT,
    иначе он не знает, чем заполнить существующие строки.
    """
    if column.nullable:
        return ""
    zero = _zero_value(column)
    if zero is None:
        # Тип, для которого нейтрального значения нет. Добавляем колонку
        # без NOT NULL: пустить приложение важнее строгости схемы.
        return ""
    return " NOT NULL DEFAULT ''" if zero == "" else " NOT NULL DEFAULT 0"


def get_session() -> Iterator[DBSession]:
    with DBSession(engine) as session:
        yield session
