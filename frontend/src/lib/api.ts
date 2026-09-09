/** Клиент REST API. Типы повторяют backend/app/schemas.py. */

export interface Word {
  w: string;
  s: number;
  e: number;
  p: number;
}

export interface Speaker {
  id: number;
  label: string;
  display_name: string;
  color: string;
  order: number;
}

export interface Segment {
  id: number;
  idx: number;
  start: number;
  end: number;
  speaker_id: number | null;
  text: string;
  words: Word[];
  edited: boolean;
}

export type SessionStatus =
  | "recording"
  | "queued"
  | "processing"
  | "ready"
  | "failed";

export interface SessionBrief {
  id: string;
  title: string;
  source_type: "live" | "upload";
  status: SessionStatus;
  quality: "draft" | "final";
  created_at: string;
  duration_sec: number;
  language: string | null;
  speaker_count: number;
  error: string | null;
  /** Лента разговора: доля времени -> порядковый номер говорящего, -1 = тишина. */
  ribbon: number[];
  /** Цвета говорящих по тому же порядковому номеру. */
  speaker_colors: string[];
}

export interface SessionDetail extends SessionBrief {
  audio_url: string | null;
  /** Затравка для модели: имена, термины, названия. */
  prompt: string;
  /** Ожидаемое число говорящих, 0 — определять автоматически. */
  num_speakers: number;
  original_filename: string | null;
  model_info: Record<string, unknown>;
  speakers: Speaker[];
  segments: Segment[];
}

export interface SpeakerStats {
  speaker_id: number | null;
  display_name: string;
  color: string;
  talk_time_sec: number;
  share: number;
  word_count: number;
  words_per_min: number;
  turns: number;
}

export interface SessionStats {
  duration_sec: number;
  speech_time_sec: number;
  word_count: number;
  speakers: SpeakerStats[];
}

export interface ExportFormat {
  id: string;
  label: string;
  ext: string;
}

export interface AppConfig {
  batch_asr_model: string;
  live_asr_model: string;
  compute_type: string;
  default_language: string;
  diarization_model: string;
  live_diarization: string;
  num_speakers: number;
  max_speakers: number;
  max_live_sessions: number;
  /** Обработка микрофона браузером. По умолчанию выключена. */
  mic_processing: boolean;
  auto_refine_live: boolean;
  formats: ExportFormat[];
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    // FastAPI кладёт человекочитаемое сообщение в detail —
    // показываем именно его, а не «500 Internal Server Error».
    let message = `Запрос не прошёл (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") message = body.detail;
    } catch {
      /* тело не JSON — оставляем общее сообщение */
    }
    throw new ApiError(message, response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  config: () => request<AppConfig>("/api/config"),

  listSessions: () => request<SessionBrief[]>("/api/sessions"),

  getSession: (id: string) => request<SessionDetail>(`/api/sessions/${id}`),

  getStats: (id: string) => request<SessionStats>(`/api/sessions/${id}/stats`),

  renameSession: (id: string, title: string) =>
    request<SessionBrief>(`/api/sessions/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),

  updateSession: (
    id: string,
    patch: { prompt?: string; num_speakers?: number },
  ) =>
    request<SessionBrief>(`/api/sessions/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),

  deleteSession: (id: string) =>
    request<void>(`/api/sessions/${id}`, { method: "DELETE" }),

  refineSession: (id: string) =>
    request<SessionBrief>(`/api/sessions/${id}/refine`, { method: "POST" }),

  updateSegment: (id: number, patch: { text?: string; speaker_id?: number }) =>
    request<Segment>(`/api/segments/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),

  updateSpeaker: (id: number, patch: { display_name?: string; color?: string }) =>
    request<Speaker>(`/api/speakers/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),

  exportUrl: (id: string, format: string) =>
    `/api/sessions/${id}/export?format=${encodeURIComponent(format)}`,

  /** Загрузка с прогрессом: fetch его не отдаёт, поэтому XHR. */
  upload(
    file: File,
    options: {
      title?: string;
      language?: string;
      prompt?: string;
      numSpeakers?: number;
      onProgress?: (fraction: number) => void;
    } = {},
  ): Promise<SessionBrief> {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      if (options.title) form.append("title", options.title);
      if (options.language) form.append("language", options.language);
      if (options.prompt) form.append("prompt", options.prompt);
      if (options.numSpeakers) form.append("num_speakers", String(options.numSpeakers));

      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/uploads");
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          options.onProgress?.(event.loaded / event.total);
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText) as SessionBrief);
        } else {
          let message = `Не удалось загрузить файл (${xhr.status})`;
          try {
            const detail = JSON.parse(xhr.responseText)?.detail;
            if (typeof detail === "string") message = detail;
          } catch {
            /* тело не JSON */
          }
          reject(new ApiError(message, xhr.status));
        }
      };
      xhr.onerror = () => reject(new ApiError("Сеть недоступна", 0));
      xhr.send(form);
    });
  },
};
