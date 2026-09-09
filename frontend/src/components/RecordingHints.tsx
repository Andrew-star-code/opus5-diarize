import { useEffect, useState } from "react";

/**
 * Две подсказки модели: словарь и число говорящих.
 *
 * Обе делают одно — сужают пространство догадок. Имена и термины модель
 * без подсказки записывает на слух («Альтрон» превращается в «трон»), а
 * не зная числа участников, диаризация перебирает диапазон от одного до
 * восьми и чаще ошибается.
 *
 * Поля намеренно необязательные и не бросаются в глаза: в большинстве
 * случаев они не нужны, а когда нужны — человек уже видит, что именно
 * распозналось не так.
 */

export interface Hints {
  prompt: string;
  numSpeakers: number;
}

interface RecordingHintsProps {
  value: Hints;
  onChange: (next: Hints) => void;
  /** Заголовок раскрывающегося блока. */
  summary?: string;
  disabled?: boolean;
}

export function RecordingHints({
  value,
  onChange,
  summary = "Подсказать модели",
  disabled = false,
}: RecordingHintsProps) {
  const [open, setOpen] = useState(false);
  const filled = Boolean(value.prompt.trim()) || value.numSpeakers > 0;

  // Если подсказки уже заданы, показываем их сразу: скрывать заполненное
  // поле — верный способ забыть, что оно влияет на результат.
  useEffect(() => {
    if (filled) setOpen(true);
  }, [filled]);

  return (
    <div className="text-sm">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="btn btn-ghost h-8 px-2 text-sm text-mist"
        aria-expanded={open}
      >
        <svg
          width="9"
          height="6"
          viewBox="0 0 9 6"
          fill="none"
          aria-hidden="true"
          style={{ transform: open ? "rotate(180deg)" : undefined }}
        >
          <path d="M1 1.5 4.5 5 8 1.5" stroke="currentColor" strokeWidth="1.3" />
        </svg>
        {summary}
        {filled && !open && <span className="text-graphite">— задано</span>}
      </button>

      {open && (
        <div className="mt-2 grid gap-3 sm:grid-cols-[1fr_auto]">
          <label className="block">
            <span className="eyebrow">Словарь</span>
            <textarea
              value={value.prompt}
              onChange={(e) => onChange({ ...value, prompt: e.target.value })}
              disabled={disabled}
              rows={2}
              maxLength={200}
              placeholder="Антон, Ольга, Сергей, Альтрон, кубернетес"
              className="mt-1 w-full resize-y rounded-md border border-rule bg-card px-3 py-2 text-sm"
            />
            <span className="mt-1 block text-[11px] text-mist">
              Имена и термины через запятую, коротким списком. Модель
              сверяется с ним, когда слышит незнакомое слово. Не пишите
              сюда описание встречи: длинный текст подмешивается в каждый
              фрагмент и вытесняет саму речь — на проверке это стоило
              трети распознанных слов.
            </span>
          </label>

          <label className="block">
            <span className="eyebrow">Говорящих</span>
            <input
              type="number"
              min={0}
              max={20}
              value={value.numSpeakers || ""}
              onChange={(e) =>
                onChange({ ...value, numSpeakers: Number(e.target.value) || 0 })
              }
              disabled={disabled}
              placeholder="авто"
              className="field mt-1 w-24"
            />
            <span className="mt-1 block text-[11px] text-mist">
              Точное число
              <br />
              улучшает разбивку
            </span>
          </label>
        </div>
      )}
    </div>
  );
}
