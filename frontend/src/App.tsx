import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { api, type Health } from "./api";
import { ModelPicker } from "./ModelPicker";
import Dashboard from "./pages/Dashboard";
import Answer from "./pages/Answer";
import Research from "./pages/Research";
import PipelinePage from "./pages/Pipeline";
import Retrieval from "./pages/Retrieval";
import Chunks from "./pages/Chunks";
import Embeddings from "./pages/Embeddings";
import Agents from "./pages/Agents";
import Knowledge from "./pages/Knowledge";
import Documents from "./pages/Documents";
import Compare from "./pages/Compare";
import Guardrails from "./pages/Guardrails";
import System from "./pages/System";

interface NavEntry {
  to: string;
  label: string;
  count?: (h: Health | null) => number | undefined;
}

/**
 * Navigation is split deliberately.
 *
 * "For you" is the practising lawyer's workspace: ask, read the answer, manage documents, browse
 * the library. "Developer / evaluation" is everything that exposes the machinery, which is what
 * the academic brief requires but which would only clutter a practitioner's day-to-day view.
 */
const NAV: { group: string; hint?: string; items: NavEntry[] }[] = [
  {
    group: "For you",
    items: [
      { to: "/dashboard", label: "Home" },
      { to: "/research", label: "Ask LegalAId" },
      {
        to: "/documents",
        label: "My documents",
        count: (h) => h?.counts.documents,
      },
      {
        to: "/knowledge",
        label: "Law library",
        count: (h) => (h ? h.counts.provisions + h.counts.cases : undefined),
      },
    ],
  },
  {
    group: "Developer / evaluation",
    hint: "How every answer was produced",
    items: [
      { to: "/pipeline", label: "Pipeline trace", count: (h) => h?.counts.traces },
      { to: "/agents", label: "Agents", count: (h) => h?.counts.agent_runs },
      { to: "/retrieval", label: "Retrieval inspector" },
      { to: "/chunks", label: "Chunk explorer", count: (h) => h?.counts.chunks },
      { to: "/embeddings", label: "Embeddings", count: (h) => h?.counts.embeddings },
      { to: "/compare", label: "RAG on / off" },
      { to: "/guardrails", label: "Guardrails" },
      { to: "/system", label: "System & models" },
    ],
  },
];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">(
    () => (localStorage.getItem("legalaid-theme") as "dark" | "light") ?? "dark",
  );
  const [model, setModel] = useState<string | null>(
    () => localStorage.getItem("legalaid-model") ?? null,
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("legalaid-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (model) localStorage.setItem("legalaid-model", model);
  }, [model]);

  useEffect(() => {
    const load = () => api.health().then(setHealth).catch(() => setHealth(null));
    load();
    const timer = setInterval(load, 20000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          LegalAId
          <small>Indian legal research workspace</small>
        </div>
        <div style={{ flex: 1 }} />
        <ModelPicker value={model} onChange={setModel} />
        <span
          className={`badge ${health?.ollama.reachable ? "ok" : "err"}`}
          title={health?.ollama.error ?? undefined}
        >
          {health?.ollama.reachable ? "Ollama online" : "Ollama offline"}
        </span>
        <button
          className="ghost small"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title="Toggle theme"
        >
          {theme === "dark" ? "Light" : "Dark"}
        </button>
      </header>

      <nav className="sidebar">
        {NAV.map((group) => (
          <div key={group.group}>
            <div className="nav-group-label">
              {group.group}
              {group.hint && <em>{group.hint}</em>}
            </div>
            {group.items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
              >
                <span className="text">{item.label}</span>
                {item.count?.(health) !== undefined && (
                  <span className="count">{item.count(health)}</span>
                )}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <main className="main">
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard health={health} />} />
          <Route path="/research" element={<Research model={model} />} />
          <Route path="/answer/:traceId" element={<Answer />} />
          <Route path="/pipeline" element={<PipelinePage />} />
          <Route path="/pipeline/:traceId" element={<PipelinePage />} />
          <Route path="/retrieval" element={<Retrieval />} />
          <Route path="/chunks" element={<Chunks />} />
          <Route path="/chunks/:chunkId" element={<Chunks />} />
          <Route path="/embeddings" element={<Embeddings />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/knowledge" element={<Knowledge />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/compare" element={<Compare model={model} />} />
          <Route path="/guardrails" element={<Guardrails />} />
          <Route path="/system" element={<System health={health} />} />
        </Routes>
      </main>
    </div>
  );
}
