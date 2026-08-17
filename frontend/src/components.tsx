/** Shared presentation pieces used across the workspace pages. */

import { useEffect, useState, type ReactNode } from "react";

export function Card({
  title,
  actions,
  children,
  tight,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  tight?: boolean;
}) {
  return (
    <section className="card">
      {(title || actions) && (
        <header className="card-head">
          <h2>{title}</h2>
          {actions}
        </header>
      )}
      <div className={tight ? "card-body tight" : "card-body"}>{children}</div>
    </section>
  );
}

export function Metric({
  label,
  value,
  sub,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
}) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

/** Colour-codes a collection so law / case / document context is distinguishable at a glance. */
export function CollectionBadge({ collection }: { collection: string }) {
  const kind = collection.startsWith("LEGAL")
    ? "law"
    : collection === "CASE_LAW"
      ? "case"
      : "doc";
  const label = collection.replace("LEGAL_", "").replace("_", " ").toLowerCase();
  return <span className={`badge ${kind}`}>{label}</span>;
}

export function StatusBadge({ status }: { status: string }) {
  const tone =
    status === "success" || status === "ok" || status === "indexed" || status === "accepted"
      ? "ok"
      : status === "failed" || status === "error"
        ? "err"
        : status === "skipped"
          ? "neutral"
          : "warn";
  return <span className={`badge ${tone}`}>{status}</span>;
}

/** Which retrievers found a chunk. This is the visible proof that hybrid search is real. */
export function RetrieverBadges({ retrievers }: { retrievers?: unknown }) {
  const list = Array.isArray(retrievers) ? (retrievers as string[]) : [];
  if (!list.length) return null;
  return (
    <>
      {list.map((r) => (
        <span
          key={r}
          className={`badge ${r === "dense" ? "law" : r === "keyword" ? "doc" : "graph"}`}
          title={
            r === "dense"
              ? "Found by vector similarity"
              : r === "keyword"
                ? "Found by BM25 exact-term matching"
                : "Found by graph traversal (no vectors)"
          }
        >
          {r}
        </span>
      ))}
    </>
  );
}

export function Panel({
  summary,
  children,
  open,
}: {
  summary: ReactNode;
  children: ReactNode;
  open?: boolean;
}) {
  return (
    <details className="panel" open={open}>
      <summary>{summary}</summary>
      <div className="panel-content">{children}</div>
    </details>
  );
}

export function Json({ value, maxHeight }: { value: unknown; maxHeight?: number }) {
  return (
    <pre style={maxHeight ? { maxHeight } : undefined}>{JSON.stringify(value, null, 2)}</pre>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="row" style={{ gap: 8 }}>
      <span className="spinner" />
      {label && <span style={{ color: "var(--text-dim)", fontSize: 13 }}>{label}</span>}
    </span>
  );
}

export function Disclaimer() {
  return (
    <div className="disclaimer">
      <strong>AI-generated legal research for professional review.</strong> Verify statutes,
      amendments, case status, citations and procedural requirements before relying on this
      output. LegalAId does not provide legal advice.
    </div>
  );
}

/** Compact visualisation of the first components of an embedding vector. */
export function VectorPreview({ values }: { values: number[] }) {
  if (!values?.length) return <span style={{ color: "var(--text-faint)" }}>—</span>;
  const max = Math.max(...values.map(Math.abs)) || 1;
  return (
    <div className="vector-preview" title={values.map((v) => v.toFixed(4)).join(", ")}>
      {values.map((v, i) => (
        <i
          key={i}
          style={{
            height: `${Math.max(4, (Math.abs(v) / max) * 100)}%`,
            background: v < 0 ? "var(--case)" : "var(--accent)",
          }}
        />
      ))}
    </div>
  );
}

export function SimilarityBar({ value }: { value: number }) {
  return (
    <div className="row" style={{ gap: 7 }}>
      <span className="mono" style={{ minWidth: 52 }}>
        {value.toFixed(4)}
      </span>
      <div className="bar" style={{ width: 62 }}>
        <div style={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }} />
      </div>
    </div>
  );
}

export function ms(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(1)} s`;
}

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fn()
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, loading, setData };
}

export function ErrorNote({ error }: { error: string }) {
  return (
    <div className="disclaimer" style={{ borderColor: "var(--err)" }}>
      {error}
    </div>
  );
}
