import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Живая запись с микрофона.
 *
 * Аудио идёт браузерным MediaRecorder в webm/opus кусками по 200 мс и
 * уходит бинарными кадрами в WebSocket. Сервер декодирует их ffmpeg-ом —
 * это тот же путь, что и у WhisperLiveKit, поэтому ничего не переизобретаем.
 *
 * Отдельно считаем уровень входа через AnalyserNode: без индикатора
 * человек не понимает, слышит его микрофон или нет, и узнаёт об этом
 * только по пустому транскрипту через минуту.
 */

export interface LiveLine {
  start: number;
  end: number;
  speaker: string | null;
  text: string;
}

export type LiveState =
  | "idle"
  | "starting"
  | "recording"
  | "stopping"
  | "done"
  | "error";

export interface LiveResult {
  sessionId: string;
  refine: boolean;
}

const CHUNK_MS = 200;
const FINALIZE_TIMEOUT_MS = 30_000;

function pickMimeType(): string | undefined {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    // Safari отдаёт mp4 — сервер всё равно декодирует через ffmpeg
    "audio/mp4",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}

/**
 * @param processAudio включить обработку микрофона средствами браузера
 *   (подавление эха, шумоподавление, автогромкость). По умолчанию
 *   выключено — см. пояснение у getUserMedia ниже.
 */
export function useLiveRecorder(language: string, processAudio = false) {
  const [state, setState] = useState<LiveState>("idle");
  const [lines, setLines] = useState<LiveLine[]>([]);
  const [buffer, setBuffer] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LiveResult | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const ws = useRef<WebSocket | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const audioCtx = useRef<AudioContext | null>(null);
  const raf = useRef<number>(0);
  const startedAt = useRef(0);
  const finalizeTimer = useRef<number>(0);

  // Обработчики сокета живут дольше одного рендера, а состояние в их
  // замыкании застывает на момент подписки. Держим актуальное значение
  // в ref, иначе onclose судит о происходящем по данным минутной давности.
  const stateRef = useRef<LiveState>("idle");
  const applyState = useCallback((next: LiveState) => {
    stateRef.current = next;
    setState(next);
  }, []);

  const teardown = useCallback(() => {
    cancelAnimationFrame(raf.current);
    window.clearTimeout(finalizeTimer.current);
    if (recorder.current && recorder.current.state !== "inactive") {
      recorder.current.stop();
    }
    stream.current?.getTracks().forEach((track) => track.stop());
    void audioCtx.current?.close().catch(() => undefined);
    recorder.current = null;
    stream.current = null;
    audioCtx.current = null;
    setLevel(0);
  }, []);

  useEffect(() => () => {
    teardown();
    ws.current?.close();
  }, [teardown]);

  const start = useCallback(async () => {
    if (stateRef.current === "recording" || stateRef.current === "starting") return;

    if (!window.isSecureContext) {
      setError(
        "Браузер даёт доступ к микрофону только по HTTPS. Откройте страницу " +
          "по защищённому адресу — как это настроить, написано в README.",
      );
      applyState("error");
      return;
    }

    applyState("starting");
    setError(null);
    setLines([]);
    setBuffer("");
    setElapsed(0);
    setResult(null);

    let media: MediaStream;
    try {
      media = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          // Все три обработки выключены намеренно.
          //
          // Это средства для голосовых звонков, и для расшифровки они
          // вредны. echoCancellation по своей задаче вырезает то, что
          // играет из колонок, — то есть удалённых участников созвона и
          // любую запись, которую слушают в комнате. noiseSuppression
          // режет всё непохожее на чистую речь и вместе с шумом уносит
          // тихие реплики. autoGainControl дёргает громкость, создавая
          // динамику, которой модель при обучении не видела.
          //
          // Модели лучше отдать сырой звук: с шумом она справляется
          // сама, а вырезанное до неё не восстановит уже никто.
          echoCancellation: processAudio,
          noiseSuppression: processAudio,
          autoGainControl: processAudio,
        },
      });
    } catch {
      setError(
        "Микрофон недоступен. Разрешите доступ в браузере и проверьте, " +
          "что устройство не занято другой программой.",
      );
      applyState("error");
      return;
    }
    stream.current = media;

    // Индикатор уровня входа
    const ctx = new AudioContext();
    audioCtx.current = ctx;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    ctx.createMediaStreamSource(media).connect(analyser);
    const samples = new Uint8Array(analyser.frequencyBinCount);
    const meter = () => {
      analyser.getByteTimeDomainData(samples);
      let sum = 0;
      for (const sample of samples) {
        const centred = (sample - 128) / 128;
        sum += centred * centred;
      }
      setLevel(Math.min(1, Math.sqrt(sum / samples.length) * 3.2));
      raf.current = requestAnimationFrame(meter);
    };
    meter();

    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(
      `${scheme}://${location.host}/ws/live?language=${encodeURIComponent(language)}`,
    );
    socket.binaryType = "arraybuffer";
    ws.current = socket;

    socket.onmessage = (event) => {
      let message: Record<string, unknown>;
      try {
        message = JSON.parse(String(event.data));
      } catch {
        return;
      }

      switch (message.type) {
        case "session":
          setSessionId(String(message.session_id));
          break;
        case "transcript":
          setLines((message.lines as LiveLine[]) ?? []);
          setBuffer(String(message.buffer ?? ""));
          break;
        case "finalized":
          window.clearTimeout(finalizeTimer.current);
          setResult({
            sessionId: String(message.session_id),
            refine: Boolean(message.refine),
          });
          applyState("done");
          break;
        case "error":
          setError(String(message.message ?? "Ошибка распознавания"));
          applyState("error");
          teardown();
          break;
      }
    };

    socket.onerror = () => {
      setError("Соединение с сервером распознавания потеряно.");
      applyState("error");
      teardown();
    };

    socket.onclose = () => {
      // Закрытие после «Остановить» — штатный конец. Всё остальное это
      // обрыв посреди записи, и молчать о нём нельзя.
      if (stateRef.current === "recording" || stateRef.current === "starting") {
        setError("Соединение прервано, запись остановлена.");
        applyState("error");
      }
    };

    socket.onopen = () => {
      const mimeType = pickMimeType();
      const rec = new MediaRecorder(media, mimeType ? { mimeType } : undefined);
      recorder.current = rec;
      rec.ondataavailable = (event) => {
        if (event.data.size > 0 && socket.readyState === WebSocket.OPEN) {
          void event.data.arrayBuffer().then((bytes) => socket.send(bytes));
        }
      };
      rec.start(CHUNK_MS);
      startedAt.current = performance.now();
      applyState("recording");
    };
  }, [applyState, language, processAudio, teardown]);

  const stop = useCallback(() => {
    if (stateRef.current !== "recording") return;
    applyState("stopping");
    recorder.current?.stop();
    stream.current?.getTracks().forEach((track) => track.stop());
    cancelAnimationFrame(raf.current);
    setLevel(0);

    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "stop" }));
      // Движок дораспознаёт хвост записи; если ответа нет — не держим
      // человека в подвешенном состоянии, а показываем, что сохранили.
      finalizeTimer.current = window.setTimeout(() => {
        if (stateRef.current !== "stopping") return;
        setResult((current) =>
          current ?? (sessionId ? { sessionId, refine: false } : null),
        );
        applyState("done");
      }, FINALIZE_TIMEOUT_MS);
    } else {
      applyState("done");
    }
  }, [applyState, sessionId]);

  // Таймер идёт локально: так он не дёргается от сетевых задержек.
  useEffect(() => {
    if (state !== "recording") return;
    const id = window.setInterval(
      () => setElapsed((performance.now() - startedAt.current) / 1000),
      200,
    );
    return () => window.clearInterval(id);
  }, [state]);

  return { state, lines, buffer, elapsed, level, error, result, sessionId, start, stop };
}
