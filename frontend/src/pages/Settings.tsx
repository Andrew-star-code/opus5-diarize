import { useEffect, useState } from "react";
import { api, type AppConfig } from "../lib/api";

/**
 * Настройки только показывают, чем именно всё считается.
 *
 * Менять модель из браузера нельзя намеренно: веса загружены в
 * видеопамять воркеров, и подмена на лету означала бы перезапуск сервиса
 * посреди чужой записи. Параметры живут в .env рядом с docker-compose.
 */
export default function Settings() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .config()
      .then(setConfig)
      .catch((err) => setError(err instanceof Error ? err.message : "Нет связи"));
  }, []);

  if (error) return <p className="text-signal">{error}</p>;
  if (!config) return <div className="shimmer h-64 rounded-[10px]" />;

  return (
    <div className="max-w-2xl">
      <h1 className="eyebrow">Настройки</h1>

      <section className="panel mt-4 divide-y divide-rule">
        <Row label="Модель распознавания, загрузка файлов" value={config.batch_asr_model} />
        <Row label="Модель распознавания, живая запись" value={config.live_asr_model} />
        <Row label="Точность вычислений" value={config.compute_type} />
        <Row label="Диаризация, загрузка файлов" value={config.diarization_model} />
        <Row
          label="Диаризация, живая запись"
          value={config.live_diarization === "none" ? "выключена" : config.live_diarization}
        />
        <Row
          label="Ожидаемое число говорящих"
          value={
            config.num_speakers > 0
              ? String(config.num_speakers)
              : `определять, не больше ${config.max_speakers}`
          }
        />
        <Row label="Язык по умолчанию" value={config.default_language} />
        <Row label="Одновременных живых записей" value={String(config.max_live_sessions)} />
        <Row
          label="Уточнять живую запись после остановки"
          value={config.auto_refine_live ? "да" : "нет"}
        />
      </section>

      <p className="mt-4 text-sm text-mist">
        Значения задаются в файле <code className="data">.env</code> рядом с{" "}
        <code className="data">docker-compose.yml</code>. После изменения
        выполните <code className="data">docker compose up -d</code>.
      </p>

      <h2 className="eyebrow mt-10">Доступ из локальной сети</h2>
      <div className="panel mt-3 px-4 py-4 text-sm leading-relaxed text-graphite">
        <p>
          Другие компьютеры в сети открывают сервис по тому же адресу, что и
          вы. Для живой записи браузеру нужен HTTPS: без него он не даёт
          доступ к микрофону — это ограничение самого браузера, а не сервиса.
        </p>
        <p className="mt-3">
          Сертификат выпускается скриптом{" "}
          <code className="data">scripts/setup_tls.ps1</code>. На каждой машине,
          с которой будут записывать, нужно один раз установить корневой
          сертификат — иначе браузер посчитает адрес небезопасным.
        </p>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-4 px-4 py-3">
      <span className="text-sm text-graphite">{label}</span>
      <span className="data ml-auto text-right text-sm break-all">{value}</span>
    </div>
  );
}
