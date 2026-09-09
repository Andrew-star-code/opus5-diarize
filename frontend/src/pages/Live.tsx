import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { useLiveRecorder, type LiveLine } from "../lib/live";
import { assignLiveColors } from "../lib/palette";
import { Ribbon, type Turn } from "../components/Ribbon";
import { timer } from "../lib/format";

const LANGUAGES = [
  { id: "ru", label: "Русский" },
  { id: "en", label: "English" },
  { id: "auto", label: "Определять" },
];

export default function Live() {
  const [language, setLanguage] = useState("ru");
  const [micProcessing, setMicProcessing] = useState(false);
  const navigate = useNavigate();
  const rec = useLiveRecorder(language, micProcessing);

  // Обработку микрофона задаёт сервер: она общая для всех, кто пишет,
  // и менять её на каждой машине отдельно было бы источником путаницы.
  useEffect(() => {
    api
      .config()
      .then((config) => setMicProcessing(config.mic_processing))
      .catch(() => undefined);
  }, []);
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  const palette = useMemo(
    () => assignLiveColors(rec.lines.map((line) => line.speaker)),
    [rec.lines],
  );

  const turns: Turn[] = useMemo(
    () =>
      rec.lines.map((line) => ({
        start: line.start,
        end: Math.max(line.end, line.start + 0.2),
        color: palette.color(line.speaker),
      })),
    [rec.lines, palette],
  );

  // Автопрокрутка — но только если человек сам не отлистал вверх читать.
  useEffect(() => {
    const node = scroller.current;
    if (node && pinned.current) node.scrollTop = node.scrollHeight;
  }, [rec.lines, rec.buffer]);

  useEffect(() => {
    if (rec.state === "done" && rec.result) {
      navigate(`/s/${rec.result.sessionId}`, {
        state: { refining: rec.result.refine },
      });
    }
  }, [rec.state, rec.result, navigate]);

  const recording = rec.state === "recording";
  const stopping = rec.state === "stopping";

  return (
    <div className="mx-auto max-w-3xl">
      <div className="flex items-baseline justify-between">
        <h1 className="eyebrow">Живая запись</h1>
        <label className="flex items-center gap-2 text-sm text-mist">
          Язык
          <select
            className="field"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            disabled={recording || stopping}
          >
            {LANGUAGES.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <section className="panel mt-4 px-6 py-8">
        <div className="flex flex-col items-center gap-5">
          <div className="flex items-center gap-3">
            {recording && <span className="rec-dot" aria-hidden="true" />}
            <span
              className="data text-[44px] leading-none font-medium tabular-nums"
              aria-label="Длительность записи"
            >
              {timer(rec.elapsed)}
            </span>
          </div>

          <LevelMeter level={rec.level} active={recording} />

          {recording || stopping ? (
            <button
              className="btn btn-solid h-11 px-6"
              onClick={rec.stop}
              disabled={stopping}
            >
              {stopping ? "Сохраняю…" : "Остановить"}
            </button>
          ) : (
            <button
              className="btn btn-signal h-11 px-6"
              onClick={() => void rec.start()}
              disabled={rec.state === "starting"}
            >
              {rec.state === "starting" ? "Подключаюсь…" : "Начать запись"}
            </button>
          )}

          {rec.state === "idle" && (
            <p className="max-w-md text-center text-sm text-mist">
              Звук идёт на ваш сервер и никуда больше. По окончании запись
              автоматически переобрабатывается крупной моделью — текст и
              разбивка по голосам станут точнее.
            </p>
          )}
        </div>

        {rec.error && (
          <div className="mt-6 rounded-[10px] border border-signal/40 px-4 py-3 text-sm text-signal">
            {rec.error}
          </div>
        )}
      </section>

      {(recording || stopping || rec.lines.length > 0) && (
        <>
          <div className="mt-6">
            <Ribbon
              turns={turns}
              duration={Math.max(rec.elapsed, 1)}
              height={12}
              label="Кто говорит сейчас"
            />
            <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1">
              {Array.from({ length: palette.count }).map((_, index) => (
                <span key={index} className="flex items-center gap-1.5">
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ background: palette.color(`SPEAKER_${String(index).padStart(2, "0")}`) }}
                  />
                  <span className="slug text-mist">Спикер {index + 1}</span>
                </span>
              ))}
            </div>
          </div>

          <div
            ref={scroller}
            onScroll={(e) => {
              const node = e.currentTarget;
              pinned.current =
                node.scrollHeight - node.scrollTop - node.clientHeight < 60;
            }}
            className="mt-5 max-h-[52vh] overflow-y-auto pr-2"
          >
            <Stream lines={rec.lines} buffer={rec.buffer} palette={palette} />
          </div>
        </>
      )}

      {rec.state === "idle" && rec.lines.length === 0 && (
        <p className="mt-8 text-center text-sm text-mist">
          Уже записанное лежит в{" "}
          <Link to="/" className="underline underline-offset-2">
            списке записей
          </Link>
          .
        </p>
      )}
    </div>
  );
}

function LevelMeter({ level, active }: { level: number; active: boolean }) {
  const bars = 28;
  const lit = Math.round(level * bars);
  return (
    <div className="flex h-4 items-end gap-[3px]" aria-hidden="true">
      {Array.from({ length: bars }).map((_, index) => {
        const on = active && index < lit;
        return (
          <span
            key={index}
            className="w-[3px] rounded-[1px] transition-[height,background-color] duration-75"
            style={{
              height: on ? `${6 + (index / bars) * 10}px` : "4px",
              background: on ? "var(--color-ink)" : "var(--color-rule)",
            }}
          />
        );
      })}
    </div>
  );
}

function Stream({
  lines,
  buffer,
  palette,
}: {
  lines: LiveLine[];
  buffer: string;
  palette: ReturnType<typeof assignLiveColors>;
}) {
  return (
    <div className="space-y-5">
      {lines.map((line, index) => {
        const newSpeaker = index === 0 || lines[index - 1].speaker !== line.speaker;
        return (
          <div key={index}>
            {newSpeaker && (
              <div className="mb-1 flex items-center gap-2">
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ background: palette.color(line.speaker) }}
                />
                <span className="slug" style={{ color: palette.color(line.speaker) }}>
                  {palette.name(line.speaker)}
                </span>
              </div>
            )}
            <p className="speech">{line.text}</p>
          </div>
        );
      })}

      {buffer && (
        // Гипотеза движка: текст ещё может быть переписан, поэтому он
        // выглядит иначе, чем уже принятые слова.
        <p className="speech speech-draft">{buffer}</p>
      )}
    </div>
  );
}
