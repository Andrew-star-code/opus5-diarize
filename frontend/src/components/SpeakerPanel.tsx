import { useEffect, useState } from "react";
import type { SessionStats, Speaker } from "../lib/api";
import { clock } from "../lib/format";

/**
 * Панель говорящих: кто сколько говорил и как их зовут на самом деле.
 *
 * Переименование здесь — не украшение: «SPEAKER_00» невозможно читать,
 * а разметка становится полезной ровно в тот момент, когда у голосов
 * появляются имена.
 */

interface SpeakerPanelProps {
  speakers: Speaker[];
  stats: SessionStats | null;
  onRename: (speakerId: number, name: string) => void;
}

export function SpeakerPanel({ speakers, stats, onRename }: SpeakerPanelProps) {
  if (speakers.length === 0) return null;

  const rows = stats?.speakers ?? [];

  return (
    <div className="panel px-4 py-4">
      <h2 className="eyebrow">Голоса</h2>
      <ul className="mt-3 space-y-3">
        {speakers.map((speaker) => {
          const row = rows.find((r) => r.speaker_id === speaker.id);
          return (
            <li key={speaker.id}>
              <div className="flex items-center gap-2">
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ background: speaker.color }}
                  aria-hidden="true"
                />
                <NameField speaker={speaker} onRename={onRename} />
                {row && (
                  <span className="data ml-auto shrink-0 text-xs text-mist">
                    {clock(row.talk_time_sec)}
                  </span>
                )}
              </div>

              {speaker.suggested_name && speaker.suggested_name !== speaker.display_name && (
                <p className="mt-1 ml-[18px] text-[11px] text-mist">
                  Похоже, это {speaker.suggested_name} ·{" "}
                  <button
                    className="underline decoration-dotted hover:text-ink"
                    onClick={() => onRename(speaker.id, speaker.suggested_name)}
                  >
                    принять
                  </button>
                </p>
              )}

              {row && (
                <>
                  <div className="mt-1.5 ml-[18px] h-[3px] overflow-hidden rounded-full bg-sunk">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.max(2, row.share * 100)}%`,
                        background: speaker.color,
                      }}
                    />
                  </div>
                  <p className="data mt-1 ml-[18px] text-[11px] text-mist">
                    {Math.round(row.share * 100)}% · {row.word_count} сл. ·{" "}
                    {Math.round(row.words_per_min)} сл/мин
                  </p>
                </>
              )}
            </li>
          );
        })}
      </ul>

      {stats && (
        <p className="data mt-4 border-t border-rule pt-3 text-[11px] text-mist">
          Речь {clock(stats.speech_time_sec)} из {clock(stats.duration_sec)} ·{" "}
          {stats.word_count} слов
        </p>
      )}
    </div>
  );
}

function NameField({
  speaker,
  onRename,
}: {
  speaker: Speaker;
  onRename: (id: number, name: string) => void;
}) {
  const [value, setValue] = useState(speaker.display_name);

  useEffect(() => setValue(speaker.display_name), [speaker.display_name]);

  const commit = () => {
    const trimmed = value.trim();
    if (trimmed && trimmed !== speaker.display_name) onRename(speaker.id, trimmed);
    else setValue(speaker.display_name);
  };

  return (
    <input
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") e.currentTarget.blur();
        if (e.key === "Escape") {
          setValue(speaker.display_name);
          e.currentTarget.blur();
        }
      }}
      // Поле выглядит как текст, пока в него не зашли: панель должна
      // читаться как сводка, а не как форма.
      className="min-w-0 flex-1 rounded border border-transparent bg-transparent px-1 py-0.5 text-sm hover:border-rule focus:border-rule focus:bg-paper"
      aria-label={`Имя говорящего, сейчас ${speaker.display_name}`}
    />
  );
}
