import { useRef } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import type { Segment, Speaker } from "../lib/api";

/**
 * Лента разговора — главный элемент интерфейса.
 *
 * Полоса времени, на которой каждая реплика окрашена в цвет своего
 * говорящего. По ней сразу видно форму разговора: монолог, интервью,
 * планёрка с перебиваниями — всё выглядит по-разному ещё до чтения
 * текста. Один и тот же компонент работает в трёх местах: миниатюрой
 * в списке записей, во всю ширину в редакторе и растущей вживую
 * во время записи.
 */

export interface Turn {
  start: number;
  end: number;
  color: string;
}

export function segmentsToTurns(segments: Segment[], speakers: Speaker[]): Turn[] {
  const colors = new Map(speakers.map((s) => [s.id, s.color]));
  return segments.map((segment) => ({
    start: segment.start,
    end: segment.end,
    color: (segment.speaker_id !== null && colors.get(segment.speaker_id)) || "#8b9499",
  }));
}

/**
 * Разворачивает сжатую ленту из списка записей обратно в отрезки.
 * Соседние доли одного говорящего склеиваются, поэтому в SVG попадает
 * несколько десятков прямоугольников, а не 240.
 */
export function bucketsToTurns(
  buckets: number[],
  colors: string[],
  duration: number,
): Turn[] {
  if (!buckets.length || duration <= 0) return [];
  const slot = duration / buckets.length;
  const turns: Turn[] = [];
  let runStart = 0;

  for (let i = 1; i <= buckets.length; i += 1) {
    const ended = i === buckets.length || buckets[i] !== buckets[runStart];
    if (!ended) continue;
    const order = buckets[runStart];
    if (order >= 0) {
      turns.push({
        start: runStart * slot,
        end: i * slot,
        color: colors[order] ?? "#8b9499",
      });
    }
    runStart = i;
  }
  return turns;
}

interface RibbonProps {
  turns: Turn[];
  duration: number;
  height?: number;
  currentTime?: number;
  onSeek?: (seconds: number) => void;
  /** Подпись для скринридера: лента сама по себе ничего не говорит. */
  label?: string;
  className?: string;
}

const VIEWBOX_WIDTH = 1000;

export function Ribbon({
  turns,
  duration,
  height = 10,
  currentTime,
  onSeek,
  label = "Лента разговора",
  className = "",
}: RibbonProps) {
  const ref = useRef<HTMLDivElement>(null);
  const span = Math.max(duration, 0.001);
  const interactive = Boolean(onSeek);

  const seekFromClientX = (clientX: number) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box || !onSeek) return;
    const fraction = Math.min(1, Math.max(0, (clientX - box.left) / box.width));
    onSeek(fraction * span);
  };

  const onKeyDown = (event: ReactKeyboardEvent) => {
    if (!onSeek || currentTime === undefined) return;
    const step = event.shiftKey ? 30 : 5;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      onSeek(Math.max(0, currentTime - step));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      onSeek(Math.min(span, currentTime + step));
    }
  };

  return (
    <div
      ref={ref}
      className={`relative w-full ${interactive ? "cursor-pointer" : ""} ${className}`}
      style={{ height }}
      onClick={interactive ? (e) => seekFromClientX(e.clientX) : undefined}
      onKeyDown={interactive ? onKeyDown : undefined}
      role={interactive ? "slider" : "img"}
      aria-label={label}
      aria-valuemin={interactive ? 0 : undefined}
      aria-valuemax={interactive ? Math.round(span) : undefined}
      aria-valuenow={interactive && currentTime !== undefined ? Math.round(currentTime) : undefined}
      tabIndex={interactive ? 0 : undefined}
    >
      <svg
        width="100%"
        height={height}
        viewBox={`0 0 ${VIEWBOX_WIDTH} ${height}`}
        preserveAspectRatio="none"
        className="block rounded-[3px] bg-sunk"
        aria-hidden="true"
      >
        {turns.map((turn, index) => {
          const x = (turn.start / span) * VIEWBOX_WIDTH;
          // Короткие вставки («ага», «да-да») иначе схлопываются в ничто,
          // а они и есть самое интересное в разговоре.
          const width = Math.max(((turn.end - turn.start) / span) * VIEWBOX_WIDTH, 1.5);
          return (
            <rect
              key={index}
              x={Math.min(x, VIEWBOX_WIDTH - width)}
              y={0}
              width={width}
              height={height}
              fill={turn.color}
            />
          );
        })}
      </svg>

      {currentTime !== undefined && (
        <div
          className="pointer-events-none absolute top-0 w-px bg-ink"
          style={{
            height,
            left: `${Math.min(100, Math.max(0, (currentTime / span) * 100))}%`,
          }}
        />
      )}
    </div>
  );
}
