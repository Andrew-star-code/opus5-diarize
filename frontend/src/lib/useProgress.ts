import { useEffect, useState } from "react";

/**
 * Подписка на прогресс обработки записи (SSE).
 *
 * Стадии приходят с сервера как есть; переводим их здесь, а не в backend,
 * потому что это текст интерфейса, а не данные.
 */

const STAGE_LABELS: Record<string, string> = {
  queued: "В очереди",
  decode: "Готовлю аудио",
  asr: "Распознаю речь",
  diarize: "Разделяю голоса",
  merge: "Собираю транскрипт",
  polish: "Сверяю реплики по смыслу",
  done: "Готово",
};

export interface Progress {
  stage: string;
  label: string;
  value: number;
  finished: boolean;
  failed: string | null;
}

export function useProgress(sessionId: string | undefined, active: boolean) {
  const [progress, setProgress] = useState<Progress | null>(null);

  useEffect(() => {
    if (!sessionId || !active) {
      setProgress(null);
      return;
    }

    const source = new EventSource(`/api/sessions/${sessionId}/events`);

    const onProgress = (event: MessageEvent) => {
      const data = JSON.parse(event.data) as { stage: string; progress: number };
      setProgress({
        stage: data.stage,
        label: STAGE_LABELS[data.stage] ?? "Обрабатываю",
        value: data.progress,
        finished: false,
        failed: null,
      });
    };

    const onDone = () => {
      setProgress({
        stage: "done",
        label: STAGE_LABELS.done,
        value: 1,
        finished: true,
        failed: null,
      });
    };

    const onFailed = (event: MessageEvent) => {
      const data = JSON.parse(event.data) as { error: string };
      setProgress({
        stage: "failed",
        label: "Не получилось",
        value: 0,
        finished: true,
        failed: data.error,
      });
    };

    source.addEventListener("progress", onProgress);
    source.addEventListener("queued", () =>
      setProgress({
        stage: "queued",
        label: STAGE_LABELS.queued,
        value: 0,
        finished: false,
        failed: null,
      }),
    );
    source.addEventListener("done", onDone);
    source.addEventListener("failed", onFailed);

    return () => source.close();
  }, [sessionId, active]);

  return progress;
}
