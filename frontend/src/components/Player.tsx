import { useEffect, useImperativeHandle, useRef, useState } from "react";
import type { RefObject } from "react";
import WaveSurfer from "wavesurfer.js";
import { clock } from "../lib/format";
import { Ribbon, type Turn } from "./Ribbon";

/**
 * Плеер записи: волна сверху, лента говорящих под ней.
 *
 * Волна показывает громкость, лента — авторство. Вместе они отвечают на
 * два разных вопроса об одном отрезке времени, поэтому и стоят рядом,
 * выровненные по одной оси.
 */

export interface PlayerHandle {
  seek: (seconds: number) => void;
  toggle: () => void;
  isPlaying: () => boolean;
}

interface PlayerProps {
  src: string;
  turns: Turn[];
  duration: number;
  onTime: (seconds: number) => void;
  handleRef: RefObject<PlayerHandle | null>;
}

const SPEEDS = [0.75, 1, 1.25, 1.5, 2];

export function Player({ src, turns, duration, onTime, handleRef }: PlayerProps) {
  const container = useRef<HTMLDivElement>(null);
  const wave = useRef<WaveSurfer | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [total, setTotal] = useState(duration);
  const [speed, setSpeed] = useState(1);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!container.current) return;

    const style = getComputedStyle(document.documentElement);
    const instance = WaveSurfer.create({
      container: container.current,
      height: 56,
      waveColor: style.getPropertyValue("--color-rule").trim() || "#d2d6d8",
      progressColor: style.getPropertyValue("--color-graphite").trim() || "#565e64",
      cursorColor: style.getPropertyValue("--color-ink").trim() || "#14171a",
      cursorWidth: 1,
      barWidth: 2,
      barGap: 1,
      barRadius: 1,
      normalize: true,
      url: src,
    });
    wave.current = instance;

    instance.on("ready", () => {
      setReady(true);
      setTotal(instance.getDuration());
    });
    instance.on("timeupdate", (seconds: number) => {
      setTime(seconds);
      onTime(seconds);
    });
    instance.on("play", () => setPlaying(true));
    instance.on("pause", () => setPlaying(false));
    instance.on("finish", () => setPlaying(false));

    return () => {
      instance.destroy();
      wave.current = null;
    };
    // Пересоздаём плеер только при смене файла: перерисовка волны
    // на каждый тик времени убила бы производительность.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);

  useImperativeHandle(handleRef, () => ({
    seek: (seconds: number) => {
      const instance = wave.current;
      if (!instance) return;
      const length = instance.getDuration() || total || 1;
      instance.seekTo(Math.min(1, Math.max(0, seconds / length)));
    },
    toggle: () => void wave.current?.playPause(),
    isPlaying: () => Boolean(wave.current?.isPlaying()),
  }));

  const changeSpeed = () => {
    const next = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length];
    setSpeed(next);
    wave.current?.setPlaybackRate(next, true);
  };

  const skip = (delta: number) => wave.current?.setTime(Math.max(0, time + delta));

  return (
    <div className="panel px-4 py-3">
      <div ref={container} className={ready ? "" : "shimmer h-14 rounded"} />

      <div className="mt-2">
        <Ribbon
          turns={turns}
          duration={total || duration}
          height={12}
          currentTime={time}
          onSeek={(seconds) => wave.current?.setTime(seconds)}
          label="Кто когда говорил. Стрелками влево и вправо — перемотка"
        />
      </div>

      <div className="mt-3 flex items-center gap-2">
        <button
          className="btn btn-solid h-9 w-9 rounded-full p-0"
          onClick={() => void wave.current?.playPause()}
          aria-label={playing ? "Пауза" : "Воспроизвести"}
        >
          {playing ? <PauseIcon /> : <PlayIcon />}
        </button>

        <button className="btn btn-ghost h-9 px-2 text-xs" onClick={() => skip(-5)}>
          −5 с
        </button>
        <button className="btn btn-ghost h-9 px-2 text-xs" onClick={() => skip(15)}>
          +15 с
        </button>

        <span className="data ml-2 text-sm text-graphite">
          {clock(time)}
          <span className="text-mist"> / {clock(total || duration)}</span>
        </span>

        <button
          className="btn btn-ghost data ml-auto h-9 px-2 text-xs"
          onClick={changeSpeed}
          aria-label="Скорость воспроизведения"
        >
          {speed}×
        </button>
      </div>
    </div>
  );
}

function PlayIcon() {
  return (
    <svg width="12" height="14" viewBox="0 0 12 14" fill="currentColor" aria-hidden="true">
      <path d="M1 1.2c0-.7.8-1.1 1.4-.7l8.2 5.3c.5.4.5 1.1 0 1.5l-8.2 5.3c-.6.4-1.4 0-1.4-.7V1.2Z" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg width="12" height="14" viewBox="0 0 12 14" fill="currentColor" aria-hidden="true">
      <rect x="1" y="1" width="3.5" height="12" rx="1" />
      <rect x="7.5" y="1" width="3.5" height="12" rx="1" />
    </svg>
  );
}
