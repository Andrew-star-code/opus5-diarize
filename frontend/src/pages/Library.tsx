import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError, type SessionBrief } from "../lib/api";
import { humanDate, humanDuration, voices } from "../lib/format";
import { Ribbon, bucketsToTurns } from "../components/Ribbon";
import { RecordingHints, type Hints } from "../components/RecordingHints";

const BUSY: SessionBrief["status"][] = ["queued", "processing", "recording"];

export default function Library() {
  const [sessions, setSessions] = useState<SessionBrief[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploads, setUploads] = useState<Record<string, number>>({});
  const [dragging, setDragging] = useState(false);
  // Подсказки применяются к загружаемым здесь файлам. Живут в состоянии
  // страницы: подряд загружают обычно записи одной встречи.
  const [hints, setHints] = useState<Hints>({ prompt: "", numSpeakers: 0 });
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      setSessions(await api.listSessions());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось получить список");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Пока что-то обрабатывается, обновляем список: SSE здесь был бы
  // избыточен — на странице нет ничего, что стоит анимировать посекундно.
  const busy = sessions?.some((s) => BUSY.includes(s.status)) ?? false;
  useEffect(() => {
    if (!busy) return;
    const id = window.setInterval(load, 4000);
    return () => window.clearInterval(id);
  }, [busy, load]);

  const handleFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    for (const file of Array.from(files)) {
      const key = `${file.name}-${file.size}-${Date.now()}`;
      setUploads((prev) => ({ ...prev, [key]: 0 }));
      try {
        await api.upload(file, {
          prompt: hints.prompt.trim() || undefined,
          numSpeakers: hints.numSpeakers || undefined,
          onProgress: (fraction) =>
            setUploads((prev) => ({ ...prev, [key]: fraction })),
        });
        await load();
      } catch (err) {
        setError(
          err instanceof ApiError ? err.message : `Не удалось загрузить ${file.name}`,
        );
      } finally {
        setUploads((prev) => {
          const next = { ...prev };
          delete next[key];
          return next;
        });
      }
    }
  };

  const remove = async (session: SessionBrief) => {
    if (!confirm(`Удалить «${session.title}» вместе с аудио?`)) return;
    try {
      await api.deleteSession(session.id);
      setSessions((prev) => prev?.filter((s) => s.id !== session.id) ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось удалить запись");
    }
  };

  const pending = Object.entries(uploads);

  return (
    <div>
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void handleFiles(e.dataTransfer.files);
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-1 rounded-[10px] border border-dashed px-5 py-10 text-center transition-colors ${
          dragging ? "border-ink bg-card" : "border-rule hover:border-mist"
        }`}
      >
        <input
          ref={fileInput}
          type="file"
          accept="audio/*,video/*"
          multiple
          className="sr-only"
          onChange={(e) => {
            void handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <span className="text-[15px]">Перетащите запись сюда</span>
        <span className="text-sm text-mist">
          или нажмите, чтобы выбрать файл на диске
        </span>
      </label>

      <div className="mt-2">
        <RecordingHints value={hints} onChange={setHints} />
      </div>

      {pending.length > 0 && (
        <div className="mt-3 space-y-2">
          {pending.map(([key, fraction]) => (
            <div key={key} className="panel flex items-center gap-3 px-4 py-3">
              <span className="data text-xs text-mist">
                {Math.round(fraction * 100)}%
              </span>
              <div className="h-[3px] flex-1 overflow-hidden rounded-full bg-sunk">
                <div
                  className="h-full bg-ink transition-[width] duration-200"
                  style={{ width: `${fraction * 100}%` }}
                />
              </div>
              <span className="text-sm text-graphite">Загружается</span>
            </div>
          ))}
        </div>
      )}

      {error && (
        <div className="mt-4 rounded-[10px] border border-signal/40 px-4 py-3 text-sm text-signal">
          {error}
        </div>
      )}

      <div className="mt-10 flex items-baseline justify-between">
        <h1 className="eyebrow">Записи</h1>
        {sessions && sessions.length > 0 && (
          <span className="data text-xs text-mist">{sessions.length}</span>
        )}
      </div>

      {sessions === null ? (
        <div className="mt-4 space-y-2">
          {[0, 1, 2].map((i) => (
            <div key={i} className="panel h-[86px] px-5 py-4">
              <div className="shimmer h-3 w-40 rounded" />
              <div className="shimmer mt-4 h-[10px] w-full rounded" />
            </div>
          ))}
        </div>
      ) : sessions.length === 0 ? (
        <div className="mt-4 rounded-[10px] border border-rule px-5 py-14 text-center">
          <p className="text-graphite">Здесь пока пусто.</p>
          <p className="mt-1 text-sm text-mist">
            Загрузите файл выше или начните живую запись с микрофона.
          </p>
          <Link to="/live" className="btn btn-solid mt-6">
            Начать запись
          </Link>
        </div>
      ) : (
        <ul className="mt-4 space-y-2">
          {sessions.map((session) => (
            <SessionRow key={session.id} session={session} onDelete={remove} />
          ))}
        </ul>
      )}
    </div>
  );
}

function SessionRow({
  session,
  onDelete,
}: {
  session: SessionBrief;
  onDelete: (s: SessionBrief) => void;
}) {
  const turns = bucketsToTurns(
    session.ribbon,
    session.speaker_colors,
    session.duration_sec,
  );
  const working = session.status === "queued" || session.status === "processing";

  return (
    <li className="panel group relative">
      <Link to={`/s/${session.id}`} className="block px-5 py-4">
        <div className="flex items-baseline gap-3">
          <span className="truncate text-[15px] font-medium">{session.title}</span>
          <StatusTag session={session} />
          <span className="data ml-auto shrink-0 text-xs text-mist">
            {humanDuration(session.duration_sec)}
          </span>
        </div>

        <div className="mt-3">
          {turns.length > 0 ? (
            <Ribbon
              turns={turns}
              duration={session.duration_sec}
              height={10}
              label={`Кто когда говорил в записи «${session.title}»`}
            />
          ) : (
            <div
              className={`h-[10px] rounded-[3px] ${working ? "shimmer" : "bg-sunk"}`}
            />
          )}
        </div>

        <div className="mt-2.5 flex items-center gap-3 text-xs text-mist">
          <span>{humanDate(session.created_at)}</span>
          {session.speaker_count > 0 && <span>{voices(session.speaker_count)}</span>}
          {session.source_type === "live" && <span>с микрофона</span>}
          {session.error && (
            <span className="truncate text-signal">{session.error}</span>
          )}
        </div>
      </Link>

      <button
        onClick={() => onDelete(session)}
        className="btn btn-ghost absolute top-3 right-3 h-8 px-2 text-xs text-mist opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
        aria-label={`Удалить запись ${session.title}`}
      >
        Удалить
      </button>
    </li>
  );
}

function StatusTag({ session }: { session: SessionBrief }) {
  const base = "eyebrow shrink-0";

  if (session.status === "recording")
    return (
      <span className={`${base} flex items-center gap-1.5 text-signal`}>
        <span className="rec-dot h-1.5 w-1.5" />
        идёт запись
      </span>
    );
  if (session.status === "queued")
    return <span className={base}>в очереди</span>;
  if (session.status === "processing")
    return <span className={`${base} text-graphite`}>распознаётся</span>;
  if (session.status === "failed")
    return <span className={`${base} text-signal`}>ошибка</span>;
  if (session.quality === "draft")
    return <span className={base}>черновик</span>;
  return null;
}
