/**
 * Research: the main workspace.
 *
 * One input accepts anything - a question, a pasted FIR, an uploaded notice - and the page then
 * shows the whole pipeline as it happens: how the router classified the request, how it rewrote
 * the text that gets embedded, what was retrieved and why, what context each agent received
 * (and was denied), and what each agent returned.
 */

import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  api,
  type AgentSummary,
  type ResearchResult,
  type RetrievedChunk,
} from "../api";
import { AnswerView } from "../AnswerView";
import {
  Card,
  CollectionBadge,
  Disclaimer,
  ErrorNote,
  Json,
  Panel,
  RetrieverBadges,
  SimilarityBar,
  Spinner,
  StatusBadge,
  ms,
} from "../components";

const EXAMPLES = [
  "My client supplied goods under a written agreement and the buyer's cheque bounced. What provisions apply and are there similar Indian cases?",
  "What compensation can be claimed for loss caused by breach of contract?",
  "Is a penalty clause in a contract enforceable, and how much can be recovered?",
  "Draft a reply to a legal notice alleging breach of a supply agreement.",
];

interface LiveStage {
  stage_key: string;
  label: string;
  service: string;
  status: string;
  duration_ms?: number | null;
}

export default function Research({ model }: { model: string | null }) {
  const [input, setInput] = useState(
    // The dashboard hands a question over rather than duplicating this page's pipeline UI.
    () => sessionStorage.getItem("legalaid-pending-question") ?? "",
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [live, setLive] = useState<LiveStage[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [uploads, setUploads] = useState<string[]>([]);
  const [drafting, setDrafting] = useState<"auto" | "on" | "off">("auto");
  const fileRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    sessionStorage.removeItem("legalaid-pending-question");
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
    };
  }, []);

  async function attachFile(file: File) {
    setError(null);
    try {
      const response = await api.upload(file);
      setUploads((prev) => [
        ...prev,
        `${response.filename} — ${response.doc_class}, ${response.pages}p, ${response.chunks} chunks, ${response.vectors_indexed} vectors`,
      ]);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function attachPastedText() {
    if (input.trim().length < 200) {
      setError(
        "Paste at least a couple of hundred characters of document text to ingest it as a document.",
      );
      return;
    }
    setError(null);
    try {
      const response = await api.paste({ text: input, filename: "pasted-document.txt" });
      setUploads((prev) => [
        ...prev,
        `${response.filename} — ${response.doc_class}, ${response.chunks} chunks, ${response.vectors_indexed} vectors`,
      ]);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function run() {
    if (!input.trim()) return;
    setBusy(true);
    setError(null);
    setResult(null);
    setLive([]);
    setElapsed(0);

    const started = Date.now();
    timerRef.current = window.setInterval(
      () => setElapsed((Date.now() - started) / 1000),
      200,
    );

    // The request resolves only when the pipeline finishes, so progress is streamed separately.
    const pending = api.research({
      question: input,
      model,
      enable_drafting: drafting === "auto" ? null : drafting === "on",
    });

    // The trace id is not known until the response arrives, so poll for the newest running trace
    // and subscribe to it. This keeps the server contract simple while still giving live progress.
    const subscription: { close: (() => void) | null } = { close: null };
    const findTrace = window.setInterval(async () => {
      try {
        const { traces } = await api.traces(1);
        const candidate = traces[0];
        if (candidate && candidate.status === "running") {
          window.clearInterval(findTrace);
          subscription.close = api.streamTrace(candidate.trace_id, (event) => {
            if (event.type === "stage_started") {
              setLive((prev) => [
                ...prev,
                {
                  stage_key: event.stage_key,
                  label: event.label,
                  service: event.service,
                  status: "running",
                },
              ]);
            } else if (event.type === "stage_finished") {
              setLive((prev) =>
                prev.map((s) =>
                  s.stage_key === event.stage_key && s.status === "running"
                    ? { ...s, status: event.status, duration_ms: event.duration_ms }
                    : s,
                ),
              );
            }
          });
        }
      } catch {
        /* transient; keep polling */
      }
    }, 700);

    try {
      setResult(await pending);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      window.clearInterval(findTrace);
      subscription.close?.();
      if (timerRef.current) window.clearInterval(timerRef.current);
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <h1>Research</h1>
        <p>
          Paste a question, an FIR, a legal notice or an agreement. The intake router decides what
          work is needed and rewrites your text into the query that gets embedded; every stage
          below is recorded, not reconstructed.
        </p>
      </div>

      <Disclaimer />

      <Card
        title="Input"
        actions={
          <span className="badge neutral">{model ?? "default model"}</span>
        }
      >
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="e.g. My client supplied goods under a written agreement and the buyer refused payment. What provisions apply, and are there similar cases?"
          style={{ minHeight: 120 }}
        />
        <div className="row between" style={{ marginTop: 10 }}>
          <div className="row">
            <button className="primary" onClick={run} disabled={busy || !input.trim()}>
              {busy ? "Running pipeline…" : "Run research"}
            </button>
            <button
              className="ghost"
              onClick={() => fileRef.current?.click()}
              disabled={busy}
              title="Upload a PDF, DOCX or TXT"
            >
              Attach file
            </button>
            <button
              className="ghost"
              onClick={attachPastedText}
              disabled={busy}
              title="Ingest the text above as a document (FIR, notice, agreement)"
            >
              Ingest text as document
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".pdf,.txt,.docx,.md"
              style={{ display: "none" }}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) attachFile(file);
                event.target.value = "";
              }}
            />
          </div>
          <label className="toggle">
            Drafting
            <select
              value={drafting}
              onChange={(event) => setDrafting(event.target.value as typeof drafting)}
              style={{ width: "auto", padding: "3px 6px", fontSize: 12 }}
            >
              <option value="auto">Router decides</option>
              <option value="on">Always</option>
              <option value="off">Never</option>
            </select>
          </label>
        </div>

        {uploads.length > 0 && (
          <div style={{ marginTop: 10 }}>
            {uploads.map((u, i) => (
              <div key={i} className="badge doc" style={{ marginRight: 6 }}>
                {u}
              </div>
            ))}
          </div>
        )}

        <div className="row" style={{ marginTop: 10, gap: 6 }}>
          {EXAMPLES.map((example, index) => (
            <button
              key={index}
              className="ghost small"
              onClick={() => setInput(example)}
              disabled={busy}
            >
              Example {index + 1}
            </button>
          ))}
        </div>
      </Card>

      {error && <ErrorNote error={error} />}

      {busy && (
        <Card
          title="Live pipeline"
          actions={<span className="mono">{elapsed.toFixed(1)} s</span>}
          tight
        >
          {live.length === 0 ? (
            <div style={{ padding: 12 }}>
              <Spinner label="Starting pipeline…" />
            </div>
          ) : (
            <div className="stage-list">
              {live.map((stage, index) => (
                <div className="stage-row" key={`${stage.stage_key}-${index}`}>
                  <span className="ord">{String(index + 1).padStart(2, "0")}</span>
                  <span className="row" style={{ gap: 8 }}>
                    <span className={`stage-dot ${stage.status}`} />
                    {stage.label}
                  </span>
                  <span className="badge neutral">{stage.service}</span>
                  <span className="num">{ms(stage.duration_ms)}</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      {result && <ResultView result={result} model={model} />}
    </>
  );
}

function ResultView({
  result,
  model,
}: {
  result: ResearchResult;
  model: string | null;
}) {
  const routing = result.routing;
  const rewritten =
    routing && routing.search_query.trim() !== routing.original_input.trim();
  const [showWorking, setShowWorking] = useState(false);

  return (
    <>
      {/* The answer comes first. Everything below it is the working. */}
      {/* AnswerView renders the prose answer plus the supporting authorities and any draft. */}
      {result.answer && Object.keys(result.answer).length > 0 ? (
        <AnswerView
          answer={result.answer}
          legal={result.legal}
          cases={result.cases}
          draft={result.draft}
        />
      ) : (
        <Card title="No answer produced">
          <p style={{ margin: 0, color: "var(--text-dim)" }}>
            The pipeline completed but the synthesis stage returned nothing. Open the trace to see
            which stage failed.
          </p>
        </Card>
      )}

      <div className="row between" style={{ margin: "18px 0 12px" }}>
        <button className="ghost" onClick={() => setShowWorking(!showWorking)}>
          {showWorking ? "Hide" : "Show"} how this answer was produced
        </button>
        <span className="row" style={{ gap: 8 }}>
          <span className="mono" style={{ fontSize: 11, color: "var(--text-faint)" }}>
            {result.trace_id} · {ms(result.total_ms)} · {model ?? "default model"}
          </span>
          <Link className="btn small" to={`/pipeline/${result.trace_id}`}>
            Open full trace
          </Link>
        </span>
      </div>

      {showWorking && (
        <TechnicalView result={result} routing={routing} rewritten={!!rewritten} />
      )}
    </>
  );
}

function TechnicalView({
  result,
  routing,
  rewritten,
}: {
  result: ResearchResult;
  routing: ResearchResult["routing"];
  rewritten: boolean;
}) {
  return (
    <>
      <Card
        title="How this answer was produced"
        actions={
          <span className="row" style={{ gap: 8 }}>
            <span className="mono" style={{ color: "var(--text-faint)" }}>
              {result.trace_id}
            </span>
            <span className="badge neutral">{ms(result.total_ms)}</span>
            <Link className="btn small" to={`/pipeline/${result.trace_id}`}>
              Open full trace
            </Link>
          </span>
        }
      >
        {result.warnings.length > 0 && (
          <div className="disclaimer" style={{ borderColor: "var(--warn)" }}>
            {result.warnings.map((w, i) => (
              <div key={i}>{w}</div>
            ))}
          </div>
        )}

        {/* ---------------------------------------------- 1. routing decision */}
        {routing && (
          <Panel
            open
            summary={
              <>
                <strong>1 · Intake router</strong>
                <span className="badge law">{routing.task}</span>
                <span className="badge neutral">{routing.input_kind}</span>
                {routing.document_type !== "none" && (
                  <span className="badge doc">{routing.document_type}</span>
                )}
                {routing.used_fallback && (
                  <span className="badge warn">deterministic fallback</span>
                )}
              </>
            }
          >
            <dl className="kv">
              <dt>Your input</dt>
              <dd style={{ color: "var(--text-dim)" }}>{routing.original_input}</dd>
              <dt>Text actually embedded</dt>
              <dd>
                <div
                  className="chunk-text"
                  style={{ maxHeight: 130, borderLeftColor: "var(--accent)" }}
                >
                  {routing.search_query}
                </div>
                {rewritten ? (
                  <div style={{ marginTop: 6 }}>
                    <span className="badge ok">rewritten for retrieval</span>{" "}
                    <span style={{ color: "var(--text-faint)", fontSize: 12 }}>
                      the conversational wrapper is stripped so the vector carries legal signal
                    </span>
                  </div>
                ) : (
                  <span className="badge neutral" style={{ marginTop: 6 }}>
                    used as-is
                  </span>
                )}
              </dd>
              {routing.legal_concepts.length > 0 && (
                <>
                  <dt>Legal concepts</dt>
                  <dd>
                    {routing.legal_concepts.map((c) => (
                      <span key={c} className="badge law" style={{ marginRight: 5 }}>
                        {c}
                      </span>
                    ))}
                  </dd>
                </>
              )}
              <dt>Agents selected</dt>
              <dd>
                {routing.agents_requested.map((a) => (
                  <span key={a} className="badge ok" style={{ marginRight: 5 }}>
                    {a}
                  </span>
                ))}
              </dd>
              <dt>Reasoning</dt>
              <dd style={{ color: "var(--text-dim)" }}>{routing.reasoning}</dd>
            </dl>
          </Panel>
        )}

        {/* ---------------------------------------------- 2. retrieval */}
        {result.retrieval && (
          <Panel
            summary={
              <>
                <strong>2 · Retrieval</strong>
                <span className="badge neutral">
                  {result.retrieval.chunks.length} chunks
                </span>
                <span className="badge neutral">
                  {result.retrieval.context_tokens} ctx tokens
                </span>
                <span className="badge neutral">{ms(result.retrieval.total_ms)}</span>
              </>
            }
          >
            {result.retrieval.query_embedding && (
              <dl className="kv" style={{ marginBottom: 12 }}>
                <dt>Embedding model</dt>
                <dd className="mono">
                  {result.retrieval.query_embedding.model} ·{" "}
                  {result.retrieval.query_embedding.dimensions} dims ·{" "}
                  {ms(result.retrieval.query_embedding.latency_ms)}
                </dd>
                <dt>Query vector (first values)</dt>
                <dd className="mono" style={{ color: "var(--text-dim)" }}>
                  [{result.retrieval.query_embedding.preview.slice(0, 8).map((v) => v.toFixed(4)).join(", ")}, …]
                </dd>
              </dl>
            )}
            <table>
              <thead>
                <tr>
                  <th>Collection</th>
                  <th className="num">Candidates</th>
                  <th className="num">Top-k</th>
                  <th className="num">Returned</th>
                  <th className="num">Search</th>
                </tr>
              </thead>
              <tbody>
                {result.retrieval.searches
                  .filter((s) => s.candidate_count > 0)
                  .map((s) => (
                    <tr key={s.collection}>
                      <td>
                        <CollectionBadge collection={s.collection} />
                      </td>
                      <td className="num">{s.candidate_count}</td>
                      <td className="num">{s.top_k}</td>
                      <td className="num">{s.returned}</td>
                      <td className="num">{ms(s.search_ms)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
            <div style={{ marginTop: 12 }}>
              {result.retrieval.chunks.map((chunk) => (
                <ChunkRow key={chunk.chunk_id} chunk={chunk} />
              ))}
            </div>
          </Panel>
        )}

        {/* ---------------------------------------------- 3. agents */}
        {Object.entries(result.agents).map(([key, agent], index) => (
          <AgentPanel key={key} index={index + 3} agentKey={key} agent={agent} />
        ))}

        {/* ---------------------------------------------- 4. loop */}
        {result.loop_iterations.length > 0 && (
          <Panel
            summary={
              <>
                <strong>Verification &amp; refinement loop</strong>
                <span className="badge neutral">
                  {result.loop_iterations.length} iteration
                  {result.loop_iterations.length === 1 ? "" : "s"}
                </span>
              </>
            }
          >
            <table>
              <thead>
                <tr>
                  <th className="num">#</th>
                  <th>Agent</th>
                  <th>Action</th>
                  <th className="num">Problems</th>
                  <th>Corrective guidance</th>
                </tr>
              </thead>
              <tbody>
                {result.loop_iterations.map((it, i) => (
                  <tr key={i}>
                    <td className="num">{it.iteration}</td>
                    <td>{it.agent}</td>
                    <td>
                      <StatusBadge status={it.action} />
                    </td>
                    <td className="num">
                      {String((it.audit as any)?.problem_count ?? 0)}
                    </td>
                    <td style={{ color: "var(--text-dim)", fontSize: 12 }}>
                      {it.guidance ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        )}
      </Card>
    </>
  );
}

function ChunkRow({ chunk }: { chunk: RetrievedChunk }) {
  const kind = chunk.collection.startsWith("LEGAL")
    ? ""
    : chunk.collection === "CASE_LAW"
      ? "case"
      : "doc";
  return (
    <Panel
      summary={
        <>
          <span className="mono" style={{ color: "var(--text-faint)" }}>
            #{chunk.rank}
          </span>
          <CollectionBadge collection={chunk.collection} />
          <strong style={{ fontWeight: 550 }}>
            {chunk.section ?? chunk.heading ?? "—"}
          </strong>
          <span style={{ color: "var(--text-dim)", fontSize: 12 }}>
            {chunk.source_title}
          </span>
          <span style={{ flex: 1 }} />
          <RetrieverBadges retrievers={chunk.metadata?.retrievers} />
          <SimilarityBar value={chunk.similarity} />
        </>
      }
    >
      <div className={`chunk-text ${kind}`}>{chunk.text}</div>
      <div style={{ marginTop: 10 }}>
        <strong style={{ fontSize: 12 }}>Why this was retrieved</strong>
        <ul style={{ margin: "6px 0 0", paddingLeft: 18, color: "var(--text-dim)", fontSize: 12 }}>
          {chunk.match_reasons.map((reason, i) => (
            <li key={i}>{reason}</li>
          ))}
        </ul>
      </div>
      <div className="row" style={{ marginTop: 10, gap: 8 }}>
        <Link className="btn small" to={`/chunks/${chunk.chunk_id}`}>
          Inspect chunk
        </Link>
        {chunk.source_url && (
          <a className="btn small" href={chunk.source_url} target="_blank" rel="noreferrer">
            Official source
          </a>
        )}
        <span className="mono" style={{ color: "var(--text-faint)", fontSize: 11 }}>
          {chunk.chunk_id} · {chunk.token_count} tokens
        </span>
      </div>
    </Panel>
  );
}

function AgentPanel({
  index,
  agentKey,
  agent,
}: {
  index: number;
  agentKey: string;
  agent: AgentSummary;
}) {
  return (
    <Panel
      summary={
        <>
          <strong>
            {index} · {agent.label}
          </strong>
          <StatusBadge status={agent.status} />
          <span className="badge neutral">{agent.model}</span>
          <span className="badge neutral">{agent.prompt}</span>
          <span style={{ flex: 1 }} />
          <span className="mono" style={{ fontSize: 11, color: "var(--text-faint)" }}>
            in {agent.input_tokens ?? "?"} / out {agent.output_tokens ?? "?"} ·{" "}
            {ms(agent.latency_ms)}
            {agent.attempts > 1 ? ` · ${agent.attempts} attempts` : ""}
          </span>
        </>
      }
    >
      <div className="grid cols-2">
        <div>
          <strong style={{ fontSize: 12 }}>
            Context received ({agent.context.included_count}) ·{" "}
            {agent.context.included_tokens} tokens
          </strong>
          <div className="list-scroll" style={{ maxHeight: 210, marginTop: 6 }}>
            {agent.context.included.length === 0 ? (
              <div style={{ color: "var(--text-faint)", fontSize: 12 }}>
                No chunks — this agent works from upstream findings only.
              </div>
            ) : (
              agent.context.included.map((item) => (
                <div key={item.chunk_id} style={{ fontSize: 12, marginBottom: 5 }}>
                  <span className="badge neutral">{item.kind}</span>{" "}
                  <Link to={`/chunks/${item.chunk_id}`}>{item.label}</Link>
                  <div style={{ color: "var(--text-faint)", fontSize: 11 }}>
                    {item.reason}
                    {item.similarity != null && ` · ${item.similarity.toFixed(4)}`}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
        <div>
          <strong style={{ fontSize: 12, color: "var(--warn)" }}>
            Deliberately withheld ({agent.context.excluded_count})
          </strong>
          <div className="list-scroll" style={{ maxHeight: 210, marginTop: 6 }}>
            {agent.context.excluded.length === 0 ? (
              <div style={{ color: "var(--text-faint)", fontSize: 12 }}>Nothing excluded.</div>
            ) : (
              agent.context.excluded.map((item) => (
                <div key={item.chunk_id} style={{ fontSize: 12, marginBottom: 5 }}>
                  <span className="badge neutral">{item.kind}</span>{" "}
                  <span style={{ color: "var(--text-dim)" }}>{item.label}</span>
                  <div style={{ color: "var(--text-faint)", fontSize: 11 }}>{item.reason}</div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <Panel summary={<span>What the model saw — system prompt</span>}>
          <pre>{agent.system_prompt}</pre>
        </Panel>
        <Panel summary={<span>What the model saw — user prompt</span>}>
          <pre>{agent.user_prompt}</pre>
        </Panel>
        <Panel summary={<span>Structured output</span>}>
          <Json value={agent.output} />
        </Panel>
        <Panel summary={<span>Raw model text</span>}>
          <pre>{agent.raw_text || "(empty)"}</pre>
        </Panel>
      </div>
      {!agent.parse_ok && agent.parse_error && (
        <div className="disclaimer" style={{ borderColor: "var(--err)", marginTop: 10 }}>
          Output could not be parsed: {agent.parse_error}
        </div>
      )}
      <div style={{ marginTop: 8, fontSize: 11, color: "var(--text-faint)" }}>
        agent_run: <span className="mono">{agent.agent_run_id}</span> · key {agentKey}
      </div>
    </Panel>
  );
}
