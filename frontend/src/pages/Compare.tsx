/**
 * RAG On / Off: the same question, the same model, with and without retrieval.
 *
 * The default question is a real, reproducible failure: asked without retrieval, Code Llama
 * states that s.73 of the Indian Contract Act concerns "the sale of goods". It is actually
 * compensation for breach. With retrieval the model is grounded in the official text.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { api, type CompareResult } from "../api";
import { Card, Disclaimer, ErrorNote, Spinner, ms } from "../components";

const SUGGESTED = [
  "What does Section 73 of the Indian Contract Act deal with?",
  "What compensation can be claimed for breach of contract?",
  "Are electronic records legally recognised under Indian law?",
];

export default function Compare({ model }: { model: string | null }) {
  const [question, setQuestion] = useState(SUGGESTED[0]);
  const [data, setData] = useState<CompareResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    setData(null);
    try {
      setData(await api.compare({ question, model }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <h1>RAG On / Off</h1>
        <p>
          The same question through the same model, once from its own knowledge and once grounded
          in retrieved statutory text.
        </p>
      </div>

      <Disclaimer />

      <Card title="Question">
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          style={{ minHeight: 62 }}
        />
        <div className="row between" style={{ marginTop: 10 }}>
          <div className="row" style={{ gap: 6 }}>
            {SUGGESTED.map((suggestion, index) => (
              <button
                key={index}
                className="ghost small"
                onClick={() => setQuestion(suggestion)}
                disabled={busy}
              >
                {index + 1}
              </button>
            ))}
          </div>
          <button className="primary" onClick={run} disabled={busy}>
            {busy ? "Running both…" : "Compare"}
          </button>
        </div>
      </Card>

      {error && <ErrorNote error={error} />}
      {busy && (
        <Card>
          <Spinner label="Generating without retrieval, then with retrieval…" />
        </Card>
      )}

      {data && (
        <>
          <div className="grid cols-2">
            <Card
              title="RAG OFF — model knowledge only"
              actions={
                <span className="badge warn">
                  {ms(data.without_rag.latency_ms)} · {data.without_rag.output_tokens} tok
                </span>
              }
            >
              <div
                className="chunk-text"
                style={{ borderLeftColor: "var(--warn)", whiteSpace: "pre-wrap" }}
              >
                {data.without_rag.answer.trim() || "(empty response)"}
              </div>
              <div style={{ marginTop: 10, fontSize: 12, color: "var(--text-faint)" }}>
                No sources. Nothing here can be verified against an authority.
              </div>
            </Card>

            <Card
              title="RAG ON — grounded in retrieved statute"
              actions={
                <span className="badge ok">
                  {ms(data.with_rag.latency_ms)} · {data.with_rag.output_tokens} tok
                </span>
              }
            >
              <div
                className="chunk-text"
                style={{ borderLeftColor: "var(--ok)", whiteSpace: "pre-wrap" }}
              >
                {data.with_rag.answer.trim() || "(empty response)"}
              </div>
              <div style={{ marginTop: 10 }}>
                <strong style={{ fontSize: 12 }}>Sources supplied to the model</strong>
                <table style={{ marginTop: 5 }}>
                  <tbody>
                    {data.with_rag.sources.map((source) => (
                      <tr key={source.chunk_id}>
                        <td>
                          <span className="badge law">{source.section ?? "—"}</span>
                        </td>
                        <td style={{ fontSize: 12 }}>{source.source_title}</td>
                        <td className="num">{source.similarity.toFixed(4)}</td>
                        <td>
                          <Link
                            className="mono"
                            style={{ fontSize: 10 }}
                            to={`/chunks/${source.chunk_id}`}
                          >
                            inspect
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          <Card title="Context that was injected">
            <pre>{data.retrieved_context}</pre>
          </Card>

          <Card title="Token accounting">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th className="num">Input tokens</th>
                  <th className="num">Output tokens</th>
                  <th className="num">Latency</th>
                  <th className="num">Sources</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>RAG off</td>
                  <td className="num">{data.without_rag.input_tokens}</td>
                  <td className="num">{data.without_rag.output_tokens}</td>
                  <td className="num">{ms(data.without_rag.latency_ms)}</td>
                  <td className="num">0</td>
                </tr>
                <tr>
                  <td>RAG on</td>
                  <td className="num">{data.with_rag.input_tokens}</td>
                  <td className="num">{data.with_rag.output_tokens}</td>
                  <td className="num">{ms(data.with_rag.latency_ms)}</td>
                  <td className="num">{data.with_rag.sources.length}</td>
                </tr>
              </tbody>
            </table>
            <div style={{ marginTop: 8, fontSize: 12, color: "var(--text-dim)" }}>
              Model: <span className="mono">{data.model}</span>. Retrieval costs input tokens and
              buys verifiable grounding.
            </div>
          </Card>
        </>
      )}
    </>
  );
}
