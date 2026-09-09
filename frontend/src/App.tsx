import { NavLink, Route, Routes } from "react-router-dom";
import Library from "./pages/Library";
import Live from "./pages/Live";
import Editor from "./pages/Editor";
import Settings from "./pages/Settings";

function Nav() {
  const link = ({ isActive }: { isActive: boolean }) =>
    `px-2 py-1 rounded-md text-sm transition-colors ${
      isActive ? "text-ink" : "text-mist hover:text-graphite"
    }`;

  return (
    <header className="sticky top-0 z-20 border-b border-rule bg-paper/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-5xl items-center gap-6 px-5">
        <NavLink to="/" className="flex items-baseline gap-2">
          <span className="data text-[15px] font-medium tracking-[0.2em] uppercase">
            Scribe
          </span>
          <span className="eyebrow hidden sm:inline">локально</span>
        </NavLink>

        <nav className="flex items-center gap-1">
          <NavLink to="/" end className={link}>
            Записи
          </NavLink>
          <NavLink to="/settings" className={link}>
            Настройки
          </NavLink>
        </nav>

        <div className="ml-auto">
          <NavLink to="/live" className="btn btn-solid">
            <span className="h-2 w-2 rounded-full bg-current" aria-hidden="true" />
            Записать
          </NavLink>
        </div>
      </div>
    </header>
  );
}

export default function App() {
  return (
    <div className="min-h-dvh">
      <Nav />
      <main className="mx-auto max-w-5xl px-5 py-8">
        <Routes>
          <Route path="/" element={<Library />} />
          <Route path="/live" element={<Live />} />
          <Route path="/s/:id" element={<Editor />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  );
}

function NotFound() {
  return (
    <div className="py-24 text-center">
      <p className="eyebrow">404</p>
      <p className="mt-2 text-graphite">Такой страницы нет.</p>
      <NavLink to="/" className="btn mt-6">
        К записям
      </NavLink>
    </div>
  );
}
