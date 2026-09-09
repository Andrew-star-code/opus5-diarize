import { memo, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { Segment, Speaker } from "../lib/api";
import { clock } from "../lib/format";

/**
 * Транскрипт: одновременно текст для чтения и поверхность для правки.
 *
 * Имя говорящего набрано как слаглайн в сценарии — моноширинным капителем
 * над репликой. Так устроены бумажные расшифровки, и так глаз находит
 * смену говорящего, не вчитываясь в текст.
 */

interface TranscriptProps {
  segments: Segment[];
  speakers: Speaker[];
  currentTime: number;
  query: string;
  onSeek: (seconds: number) => void;
  onEdit: (segmentId: number, text: string) => void;
  onReassign: (segmentId: number, speakerId: number) => void;
}

export function Transcript({
  segments,
  speakers,
  currentTime,
  query,
  onSeek,
  onEdit,
  onReassign,
}: TranscriptProps) {
  const byId = new Map(speakers.map((s) => [s.id, s]));
  const activeIndex = findActive(segments, currentTime);
  const activeRef = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);

  // Ведём текст за воспроизведением, но перестаём, если человек сам
  // прокрутил в другое место — иначе невозможно прочитать что-то выше.
  useEffect(() => {
    if (!follow) return;
    activeRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [activeIndex, follow]);

  useEffect(() => {
    let timeout = 0;
    const onWheel = () => {
      setFollow(false);
      window.clearTimeout(timeout);
      timeout = window.setTimeout(() => setFollow(true), 6000);
    };
    window.addEventListener("wheel", onWheel, { passive: true });
    return () => {
      window.removeEventListener("wheel", onWheel);
      window.clearTimeout(timeout);
    };
  }, []);

  if (segments.length === 0) {
    return (
      <p className="py-16 text-center text-sm text-mist">
        Транскрипта пока нет.
      </p>
    );
  }

  return (
    <div className="space-y-1">
      {segments.map((segment, index) => {
        const speaker = segment.speaker_id ? byId.get(segment.speaker_id) : undefined;
        const previous = index > 0 ? segments[index - 1] : null;
        const startsTurn = !previous || previous.speaker_id !== segment.speaker_id;
        return (
          <div key={segment.id} ref={index === activeIndex ? activeRef : undefined}>
            <SegmentBlock
              segment={segment}
              speaker={speaker}
              speakers={speakers}
              startsTurn={startsTurn}
              active={index === activeIndex}
              currentTime={index === activeIndex ? currentTime : -1}
              query={query}
              onSeek={onSeek}
              onEdit={onEdit}
              onReassign={onReassign}
            />
          </div>
        );
      })}
    </div>
  );
}

function findActive(segments: Segment[], time: number): number {
  for (let i = 0; i < segments.length; i += 1) {
    if (time >= segments[i].start && time < segments[i].end) return i;
  }
  return -1;
}

interface SegmentProps {
  segment: Segment;
  speaker?: Speaker;
  speakers: Speaker[];
  startsTurn: boolean;
  active: boolean;
  currentTime: number;
  query: string;
  onSeek: (seconds: number) => void;
  onEdit: (segmentId: number, text: string) => void;
  onReassign: (segmentId: number, speakerId: number) => void;
}

const SegmentBlock = memo(function SegmentBlock({
  segment,
  speaker,
  speakers,
  startsTurn,
  active,
  currentTime,
  query,
  onSeek,
  onEdit,
  onReassign,
}: SegmentProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(segment.text);

  useEffect(() => setDraft(segment.text), [segment.text]);

  const commit = () => {
    setEditing(false);
    const trimmed = draft.trim();
    if (trimmed && trimmed !== segment.text) onEdit(segment.id, trimmed);
    else setDraft(segment.text);
  };

  return (
    <div className={`group -mx-3 rounded-lg px-3 py-1.5 ${active ? "bg-card" : ""}`}>
      {startsTurn && (
        <div className="mt-4 mb-1 flex items-center gap-2">
          <button
            className="slug transition-opacity hover:opacity-70"
            style={{ color: speaker?.color ?? "var(--color-mist)" }}
            onClick={() => onSeek(segment.start)}
          >
            {speaker?.display_name ?? "Говорящий"}
          </button>
          <button
            className="data text-[11px] text-mist transition-colors hover:text-graphite"
            onClick={() => onSeek(segment.start)}
          >
            {clock(segment.start)}
          </button>

          {speakers.length > 1 && (
            // Диаризация ошибается, и чаще всего — ровно на одной реплике.
            // Поэтому смена говорящего живёт прямо здесь, а не в настройках.
            <select
              className="ml-1 rounded border border-transparent bg-transparent text-[11px] text-mist opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
              value={segment.speaker_id ?? ""}
              onChange={(e) => onReassign(segment.id, Number(e.target.value))}
              aria-label="Кто это говорит"
            >
              {speakers.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.display_name}
                </option>
              ))}
            </select>
          )}
        </div>
      )}

      {editing ? (
        <textarea
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              setDraft(segment.text);
              setEditing(false);
            }
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) commit();
          }}
          rows={Math.max(2, Math.ceil(draft.length / 70))}
          className="speech w-full resize-y rounded-md border border-rule bg-card px-3 py-2"
        />
      ) : (
        <p
          className="speech"
          onDoubleClick={() => setEditing(true)}
          title="Двойной клик — исправить текст"
        >
          <Words
            segment={segment}
            currentTime={currentTime}
            query={query}
            onSeek={onSeek}
          />
        </p>
      )}
    </div>
  );
});

function Words({
  segment,
  currentTime,
  query,
  onSeek,
}: {
  segment: Segment;
  currentTime: number;
  query: string;
  onSeek: (seconds: number) => void;
}) {
  // После ручной правки пословные тайминги выброшены на сервере:
  // подсвечивать по ним было бы враньём. Показываем текст целиком.
  if (segment.words.length === 0) {
    return <Highlighted text={segment.text} query={query} />;
  }

  return (
    <>
      {segment.words.map((word, index) => {
        const active = currentTime >= word.s && currentTime < word.e;
        return (
          <span key={index}>
            <span
              className={`word ${active ? "word-active" : ""}`}
              onClick={() => onSeek(word.s)}
            >
              <Highlighted text={word.w} query={query} />
            </span>
            {index < segment.words.length - 1 ? " " : ""}
          </span>
        );
      })}
    </>
  );
}

function Highlighted({ text, query }: { text: string; query: string }) {
  if (!query || query.length < 2) return <>{text}</>;

  const lower = text.toLowerCase();
  const needle = query.toLowerCase();
  const parts: ReactNode[] = [];
  let cursor = 0;

  for (;;) {
    const hit = lower.indexOf(needle, cursor);
    if (hit === -1) break;
    if (hit > cursor) parts.push(text.slice(cursor, hit));
    parts.push(
      <mark key={hit} className="hit bg-transparent text-inherit">
        {text.slice(hit, hit + needle.length)}
      </mark>,
    );
    cursor = hit + needle.length;
  }

  if (parts.length === 0) return <>{text}</>;
  parts.push(text.slice(cursor));
  return <>{parts}</>;
}
