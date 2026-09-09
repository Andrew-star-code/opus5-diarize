import { useEffect, useRef, useState } from "react";
import { api, type ExportFormat } from "../lib/api";

/**
 * Выгрузка транскрипта. Форматы приходят с сервера, чтобы список в меню
 * не расходился с тем, что backend умеет отдать.
 */
export function ExportMenu({
  sessionId,
  formats,
  disabled,
}: {
  sessionId: string;
  formats: ExportFormat[];
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: MouseEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={root} className="relative">
      <button
        className="btn"
        onClick={() => setOpen((value) => !value)}
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        Выгрузить
        <svg width="9" height="6" viewBox="0 0 9 6" fill="none" aria-hidden="true">
          <path d="M1 1.5 4.5 5 8 1.5" stroke="currentColor" strokeWidth="1.3" />
        </svg>
      </button>

      {open && (
        <div
          role="menu"
          className="panel absolute right-0 z-30 mt-1 w-44 overflow-hidden py-1 shadow-lg"
        >
          {formats.map((format) => (
            <a
              key={format.id}
              role="menuitem"
              href={api.exportUrl(sessionId, format.id)}
              // download здесь только подсказка: имя файла всё равно
              // задаёт сервер через Content-Disposition, с кириллицей.
              download
              onClick={() => setOpen(false)}
              className="flex items-center justify-between px-3 py-1.5 text-sm hover:bg-sunk"
            >
              {format.label}
              <span className="data text-[11px] text-mist">.{format.ext}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
