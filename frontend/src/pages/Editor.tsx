import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import {
  api,
  type AppConfig,
  type SessionDetail,
  type SessionStats,
} from "../lib/api";
import { humanDate, voices } from "../lib/format";
import { useProgress } from "../lib/useProgress";
import { Player, type PlayerHandle } from "../components/Player";
import { Transcript } from "../components/Transcript";
import { SpeakerPanel } from "../components/SpeakerPanel";
import { ExportMenu } from "../components/ExportMenu";
import { segmentsToTurns } from "../components/Ribbon";
import { RecordingHints, type Hints } from "../components/RecordingHints";

export default function Editor() {
  const { id = "" } = useParams();
  const location = useLocation();
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [stats, setStats] = useState<SessionStats | null>(null);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [query, setQuery] = useState("");
  const [hints, setHints] = useState<Hints>({ prompt: "", numSpeakers: 0 });
  const player = useRef<PlayerHandle>(null);
  const search = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const [detail, statistics] = await Promise.all([
        api.getSession(id),
        api.getStats(id),
      ]);
      setSession(detail);
      setStats(statistics);
      setHints({ prompt: detail.prompt, numSpeakers: detail.num_speakers });
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Запись не открылась");
    }
  }, [id]);

  useEffect(() => {
    void load();
    void api.config().then(setConfig).catch(() => undefined);
  }, [load]);

  // Сразу после живой записи сюда приходят с флагом «идёт уточнение»,
  // и подписка на прогресс должна работать ещё до первого ответа сервера.
  const cameFromLive = Boolean((location.state as { refining?: boolean } | null)?.refining);
  const working =
    cameFromLive ||
    session?.status === "queued" ||
    session?.status === "processing";
  const progress = useProgress(id, Boolean(working));

  useEffect(() => {
    if (progress?.finished) void load();
  }, [progress?.finished, load]);

  const turns = useMemo(
    () => (session ? segmentsToTurns(session.segments, session.speakers) : []),
    [session],
  );

  const seek = useCallback((seconds: number) => {
    player.current?.seek(seconds);
    setCurrentTime(seconds);
  }, []);

  // Клавиатура: пробел — воспроизведение, Ctrl+F — поиск по транскрипту,
  // а не по странице. И то и другое ожидаемо для такого редактора.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
      if (event.key === "f" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        search.current?.focus();
        search.current?.select();
        return;
      }
      if (event.code === "Space" && !typing) {
        event.preventDefault();
        player.current?.toggle();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const rename = async (title: string) => {
    if (!session || title === session.title) return;
    setSession({ ...session, title });
    try {
      await api.renameSession(session.id, title);
    } catch {
      void load();
    }
  };

  // Колбэки обязаны быть стабильными: транскрипт на сотни реплик
  // перерисовывается по четыре раза в секунду вслед за плеером, и
  // memo на реплике спасает, только если её пропсы не меняются.
  const editSegment = useCallback(
    async (segmentId: number, text: string) => {
      // Оптимистично: правка текста — самое частое действие в редакторе,
      // и ждать ответа сервера на каждое исправление невыносимо.
      setSession((prev) =>
        prev
          ? {
              ...prev,
              segments: prev.segments.map((s) =>
                s.id === segmentId ? { ...s, text, words: [], edited: true } : s,
              ),
            }
          : prev,
      );
      try {
        await api.updateSegment(segmentId, { text });
      } catch {
        void load();
      }
    },
    [load],
  );

  const reassign = useCallback(
    async (segmentId: number, speakerId: number) => {
      setSession((prev) =>
        prev
          ? {
              ...prev,
              segments: prev.segments.map((s) =>
                s.id === segmentId
                  ? { ...s, speaker_id: speakerId, edited: true }
                  : s,
              ),
            }
          : prev,
      );
      try {
        await api.updateSegment(segmentId, { speaker_id: speakerId });
        setStats(await api.getStats(id));
      } catch {
        void load();
      }
    },
    [id, load],
  );

  const renameSpeaker = useCallback(
    async (speakerId: number, name: string) => {
      setSession((prev) =>
        prev
          ? {
              ...prev,
              speakers: prev.speakers.map((s) =>
                s.id === speakerId ? { ...s, display_name: name } : s,
              ),
            }
          : prev,
      );
      try {
        await api.updateSpeaker(speakerId, { display_name: name });
      } catch {
        void load();
      }
    },
    [load],
  );

  const refine = async () => {
    try {
      // Сначала сохраняем подсказки: воркер прочитает их из записи,
      // когда возьмёт задание, поэтому порядок здесь существенный.
      await api.updateSession(id, {
        prompt: hints.prompt.trim(),
        num_speakers: hints.numSpeakers,
      });
      await api.refineSession(id);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось поставить в очередь");
    }
  };

  if (error && !session) {
    return (
      <div className="py-20 text-center">
        <p className="text-signal">{error}</p>
        <Link to="/" className="btn mt-6">
          К записям
        </Link>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="space-y-3">
        <div className="shimmer h-6 w-64 rounded" />
        <div className="shimmer h-28 rounded-[10px]" />
        <div className="shimmer h-96 rounded-[10px]" />
      </div>
    );
  }

  return (
    <div>
      <div className="flex flex-wrap items-center gap-3">
        <Link to="/" className="btn btn-ghost h-8 px-2 text-sm text-mist">
          ← Записи
        </Link>
        {session.quality === "draft" && <span className="eyebrow">черновик</span>}
      </div>

      <div className="mt-2 flex flex-wrap items-baseline gap-x-4 gap-y-2">
        <TitleField title={session.title} onRename={rename} />
        <span className="data text-xs text-mist">
          {humanDate(session.created_at)}
          {session.speaker_count > 0 && ` · ${voices(session.speaker_count)}`}
          {session.language && ` · ${session.language}`}
        </span>

        <div className="ml-auto flex items-center gap-2">
          {session.quality === "draft" && !working && (
            <button className="btn" onClick={refine}>
              Уточнить
            </button>
          )}
          <ExportMenu
            sessionId={session.id}
            formats={config?.formats ?? []}
            disabled={session.segments.length === 0}
          />
        </div>
      </div>

      {working && <ProgressBanner progress={progress} />}

      {session.status === "failed" && session.error && (
        <div className="mt-4 rounded-[10px] border border-signal/40 px-4 py-3 text-sm">
          <p className="text-signal">Обработка не прошла.</p>
          <p className="mt-1 text-graphite">{session.error}</p>
          <button className="btn mt-3 h-8 text-sm" onClick={refine}>
            Попробовать снова
          </button>
        </div>
      )}

      {session.audio_url && (
        <div className="sticky top-14 z-10 mt-5 bg-paper pt-1 pb-2">
          <Player
            src={session.audio_url}
            turns={turns}
            duration={session.duration_sec}
            onTime={setCurrentTime}
            handleRef={player}
          />
        </div>
      )}

      <div className="mt-6 grid gap-8 lg:grid-cols-[1fr_236px]">
        <div className="min-w-0">
          <div className="mb-2 flex items-center gap-3">
            <input
              ref={search}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Найти в транскрипте"
              className="field w-full max-w-xs"
              aria-label="Поиск по транскрипту"
            />
            {query.length >= 2 && (
              <span className="data text-xs text-mist">
                {countMatches(session, query)}
              </span>
            )}
          </div>

          <Transcript
            segments={session.segments}
            speakers={session.speakers}
            currentTime={currentTime}
            query={query}
            onSeek={seek}
            onEdit={editSegment}
            onReassign={reassign}
          />
        </div>

        <aside className="lg:sticky lg:top-52 lg:self-start">
          <SpeakerPanel
            speakers={session.speakers}
            stats={stats}
            onRename={renameSpeaker}
          />
          {session.audio_url && (
            <div className="panel mt-3 px-4 py-4">
              <RecordingHints
                value={hints}
                onChange={setHints}
                summary="Подсказать модели"
                disabled={Boolean(working)}
              />
              <button
                className="btn mt-3 h-8 w-full text-sm"
                onClick={refine}
                disabled={Boolean(working)}
              >
                {working ? "Обрабатывается…" : "Распознать заново"}
              </button>
              <p className="mt-2 text-[11px] leading-relaxed text-mist">
                Заметили неверно распознанное имя или термин — впишите его
                и запустите заново. Правки текста при этом пропадут.
              </p>
            </div>
          )}

          <p className="mt-3 px-1 text-[11px] leading-relaxed text-mist">
            Двойной клик по реплике — исправить текст. Клик по слову —
            перемотать. Пробел — пуск и пауза.
          </p>
        </aside>
      </div>
    </div>
  );
}

function countMatches(session: SessionDetail, query: string): string {
  const needle = query.toLowerCase();
  let count = 0;
  for (const segment of session.segments) {
    const text = segment.text.toLowerCase();
    let from = 0;
    for (;;) {
      const hit = text.indexOf(needle, from);
      if (hit === -1) break;
      count += 1;
      from = hit + needle.length;
    }
  }
  return count === 0 ? "ничего" : `${count}`;
}

function ProgressBanner({
  progress,
}: {
  progress: ReturnType<typeof useProgress>;
}) {
  const value = progress?.value ?? 0;
  return (
    <div className="panel mt-4 px-4 py-3">
      <div className="flex items-center gap-3">
        <span className="text-sm">
          {progress?.label ?? "Ставлю в очередь"}
        </span>
        <span className="data ml-auto text-xs text-mist">
          {Math.round(value * 100)}%
        </span>
      </div>
      <div className="mt-2 h-[3px] overflow-hidden rounded-full bg-sunk">
        <div
          className="h-full bg-ink transition-[width] duration-500"
          style={{ width: `${Math.max(2, value * 100)}%` }}
        />
      </div>
      <p className="mt-2 text-[11px] text-mist">
        Черновик уже можно читать — он обновится, когда пройдёт полная
        обработка.
      </p>
    </div>
  );
}

function TitleField({
  title,
  onRename,
}: {
  title: string;
  onRename: (value: string) => void;
}) {
  const [value, setValue] = useState(title);
  useEffect(() => setValue(title), [title]);

  return (
    <input
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onBlur={() => {
        const trimmed = value.trim();
        if (trimmed) onRename(trimmed);
        else setValue(title);
      }}
      onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
      className="min-w-0 flex-1 rounded border border-transparent bg-transparent py-1 text-[22px] font-medium hover:border-rule focus:border-rule focus:bg-card"
      aria-label="Название записи"
    />
  );
}
