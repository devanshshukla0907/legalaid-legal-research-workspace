/**
 * Retrieval Inspector: run retrieval alone and see the hybrid retrievers' contributions.
 *
 * The toggles are the point. Turning keyword or graph off and re-running shows exactly what each
 * retriever adds, which is what makes the hybrid claim verifiable rather than asserted.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { api, type RetrievalResponse } from "../api";
import {
  Card,
  CollectionBadge,
  Empty,
  ErrorNote,
  Panel,
  RetrieverBadges,
  SimilarityBar,
  Spinner,
  VectorPreview,
  ms,
} from "../components";

const MODES = [
  ["COMBINED_LEGAL_RESEARCH", "Combined"],
  ["LAW_SEARCH", "Law only"],
  ["CASE_SEARCH", "Cases only"],
  ["USER_DOCUMENT_SEARCH", "My documents"],
] as const;

export default function Retrieval() {
  const [question, setQuestion] = useState("compensation for loss caused by breach of contract");
  const [mode, setMode] = useState<string>("COMBINED_LEGAL_RESEARCH");
  const [topK, setTopK] = useState(8);
  const [keyword, setKeyword] = useState(true);
  const [graph, setGraph] = useState(true);
  const [data, setData] = useState<RetrievalResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setData(
        await api.retrieve({
          question,
          mode,
          top_k: topK,
          enable_keyword: keyword,
          enable_graph: graph,
        }),
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <h1>Retrieval Inspector</h1>
        <p>
          Query embedding → source filter → vector search → metadata filter → rank → context.
          Switch the retrievers off to see what each contributes.
        </p>
      </div>

      <Card title="Query">
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          style={{ minHeight: 70 }}
        />
        <div className="row between" style={{ marginTop: 10 }}>
          <div className="row">
            <select
              value={mode}
              onChange={(event) => setMode(event.target.value)}
              style={{ width: "auto" }}
            >
              {MODES.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
            <label className="row" style={{ gap: 5 }}>
              <span style={{ fontSize: 12, color: "var(--text-faint)" }}>top-k</span>
              <input
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
                style={{ width: 62 }}
              />
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked
                disabled
                title="Dense vector search is always on"
              />
              dense (always on)
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={keyword}
                onChange={(event) => setKeyword(event.target.checked)}
              />
              keyword (BM25)
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={graph}
                onChange={(event) => setGraph(event.target.checked)}
              />
              graph (no vectors)
            </label>
          </div>
          <button className="primary" onClick={run} disabled={busy}>
            {busy ? "Searching…" : "Search"}
          </button>
        </div>
      </Card>

      {error && <ErrorNote error={error} />}
      {busy && <Card><Spinner label="Embedding query and searching…" /></Card>}

      {data && (
        <>
          {data.query_embedding && (
            <Card title="Query embedding">
              <div className="grid cols-2">
                <dl className="kv">
                  <dt>Model</dt>
                  <dd className="mono">{data.query_embedding.model}</dd>
                  <dt>Dimensions</dt>
                  <dd className="mono">{data.query_embedding.dimensions}</dd>
                  <dt>Input</dt>
                  <dd className="mono">
                    {data.query_embedding.input_chars} chars ·{" "}
                    {data.query_embedding.input_tokens} tokens
                  </dd>
                  <dt>Latency</dt>
                  <dd className="mono">{ms(data.query_embedding.latency_ms)}</dd>
                </dl>
                <div>
                  <div style={{ fontSize: 11, color: "var(--text-faint)", marginBottom: 5 }}>
                    First {data.query_embedding.preview.length} components
                  </div>
                  <VectorPreview values={data.query_embedding.preview} />
                  <div className="mono" style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 6 }}>
                    [{data.query_embedding.preview.map((v) => v.toFixed(4)).join(", ")}, …]
                  </div>
                </div>
              </div>
            </Card>
          )}

          <Card title="Searches per collection">
            <table>
              <thead>
                <tr>
                  <th>Collection</th>
                  <th className="num">Candidates</th>
                  <th className="num">Eligible after filter</th>
                  <th className="num">Top-k</th>
                  <th className="num">Returned</th>
                  <th className="num">Search</th>
                  <th className="num">Rank</th>
                </tr>
              </thead>
              <tbody>
                {data.searches.map((s) => (
                  <tr key={s.collection}>
                    <td>
                      <CollectionBadge collection={s.collection} />
                    </td>
                    <td className="num">{s.candidate_count}</td>
                    <td className="num">{s.filtered_count}</td>
                    <td className="num">{s.top_k}</td>
                    <td className="num">{s.returned}</td>
                    <td className="num">{ms(s.search_ms)}</td>
                    <td className="num">{ms(s.rank_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card
            title={`Results (${data.chunks.length})`}
            actions={
              <span className="badge neutral">
                {data.context_tokens} context tokens · {ms(data.total_ms)}
              </span>
            }
          >
            {data.chunks.length === 0 ? (
              <Empty>No chunks passed the similarity threshold.</Empty>
            ) : (
              data.chunks.map((chunk) => (
                <Panel
                  key={chunk.chunk_id}
                  summary={
                    <>
                      <span className="mono" style={{ color: "var(--text-faint)" }}>
                        #{chunk.rank}
                      </span>
                      <CollectionBadge collection={chunk.collection} />
                      <strong style={{ fontWeight: 550 }}>
                        {chunk.section ?? chunk.heading ?? "—"}
                      </strong>
                      <span style={{ fontSize: 12, color: "var(--text-dim)" }}>
                        {chunk.source_title}
                      </span>
                      <span style={{ flex: 1 }} />
                      <RetrieverBadges retrievers={chunk.metadata?.retrievers} />
                      <SimilarityBar value={chunk.similarity} />
                    </>
                  }
                >
                  <div
                    className={`chunk-text ${chunk.collection === "CASE_LAW" ? "case" : chunk.collection === "USER_DOCUMENTS" ? "doc" : ""}`}
                  >
                    {chunk.text}
                  </div>
                  <div style={{ marginTop: 10, fontSize: 12 }}>
                    <strong>Why this was retrieved</strong>
                    <ul style={{ margin: "5px 0 0", paddingLeft: 18, color: "var(--text-dim)" }}>
                      {chunk.match_reasons.map((reason, i) => (
                        <li key={i}>{reason}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="row" style={{ marginTop: 10 }}>
                    <Link className="btn small" to={`/chunks/${chunk.chunk_id}`}>
                      Inspect chunk
                    </Link>
                    {chunk.source_url && (
                      <a
                        className="btn small"
                        href={chunk.source_url}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Official source
                      </a>
                    )}
                  </div>
                </Panel>
              ))
            )}
          </Card>
        </>
      )}
    </>
  );
}
