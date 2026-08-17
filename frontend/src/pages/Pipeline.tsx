/** Pipeline Trace: every recorded stage of a request, as a node graph and an expandable list. */

import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type TraceDetail, type TraceStage } from "../api";
import {
  Card,
  CollectionBadge,
  Empty,
  ErrorNote,
  Json,
  Panel,
  SimilarityBar,
  Spinner,
  StatusBadge,
  ms,
  useAsync,
} from "../components";

export default function PipelinePage() {
  const { traceId } = useParams();
  const list = useAsync(() => api.traces(40), []);
  const [selected, setSelected] = useState<string | null>(traceId ?? null);
  const active = selected ?? traceId ?? list.data?.traces?.[0]?.trace_id ?? null;

  return (
    <>
      <div className="page-head">
        <h1>Pipeline Trace</h1>
        <p>
          Every request creates a trace. Each stage records its service, transport, duration,
          input and output, so the whole journey from user input to final response is inspectable.
        </p>
      </div>

      <div className="split">
        <Card title="Traces" tight>
          {list.loading ? (
            <div style={{ padding: 12 }}>
              <Spinner />
            </div>
          ) : list.error ? (
            <ErrorNote error={list.error} />
          ) : !list.data?.traces?.length ? (
            <Empty>No traces yet. Run a research query.</Empty>
          ) : (
            <div className="list-scroll">
              <table>
                <tbody>
                  {list.data.traces.map((trace: any) => (
                    <tr
                      key={trace.trace_id}
                      className="clickable"
                      onClick={() => setSelected(trace.trace_id)}
                      style={
                        trace.trace_id === active
                          ? { background: "var(--bg-hover)" }
                          : undefined
                      }
                    >
                      <td>
                        <div className="row" style={{ gap: 6 }}>
                          <StatusBadge status={trace.status} />
                          <span className="mono" style={{ fontSize: 11 }}>
                            {trace.trace_id}
                          </span>
                        </div>
                        <div
                          style={{
                            fontSize: 12,
                            color: "var(--text-dim)",
                            marginTop: 3,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                            maxWidth: 300,
                          }}
                        >
                          {trace.question}
                        </div>
                        <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
                          {trace.stages} stages · {ms(trace.total_ms)}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <div>{active ? <TraceView traceId={active} /> : <Empty>Select a trace.</Empty>}</div>
      </div>
    </>
  );
}

function TraceView({ traceId }: { traceId: string }) {
  const { data, error, loading } = useAsync<TraceDetail>(() => api.trace(traceId), [traceId]);

  if (loading) return <Card><Spinner label="Loading trace…" /></Card>;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  return (
    <>
      <Card
        title={`Trace ${data.trace_id}`}
        actions={
          <span className="row" style={{ gap: 8 }}>
            <StatusBadge status={data.status} />
            <span className="badge neutral">{ms(data.total_ms)}</span>
          </span>
        }
      >
        <dl className="kv">
          <dt>Question</dt>
          <dd>{data.question}</dd>
          <dt>Mode</dt>
          <dd className="mono">{data.mode}</dd>
          <dt>Started</dt>
          <dd className="mono">{data.started_at ?? "—"}</dd>
          <dt>Stages / agents / model calls</dt>
          <dd className="mono">
            {data.stages.length} / {data.agent_runs.length} / {data.model_runs.length}
          </dd>
        </dl>
      </Card>

      <Card title="Node graph">
        <NodeGraph stages={data.stages} />
      </Card>

      <Card title="Stages" tight>
        <div className="stage-list">
          {data.stages.map((stage) => (
            <StageEntry key={stage.id} stage={stage} />
          ))}
        </div>
      </Card>

      {data.retrievals.length > 0 && (
        <Card title="Retrieval detail">
          {data.retrievals.map((run) => (
            <Panel
              key={run.id}
              summary={
                <>
                  <CollectionBadge collection={run.collection} />
                  <span className="mono" style={{ fontSize: 12 }}>
                    {run.candidate_count} candidates → top {run.top_k} → {run.returned_count}
                  </span>
                  <span style={{ flex: 1 }} />
                  <span className="badge neutral">{ms(run.search_ms)}</span>
                </>
              }
            >
              <dl className="kv" style={{ marginBottom: 10 }}>
                <dt>Query text embedded</dt>
                <dd>{run.query_text}</dd>
                <dt>Embedding</dt>
                <dd className="mono">
                  {run.embedding_model} · {run.dimensions} dims · {ms(run.embedding_ms)}
                </dd>
                <dt>Query vector</dt>
                <dd className="mono" style={{ color: "var(--text-dim)" }}>
                  [{run.query_vector_preview.slice(0, 8).map((v) => v.toFixed(4)).join(", ")}, …]
                </dd>
                <dt>Min similarity</dt>
                <dd className="mono">{run.min_similarity}</dd>
              </dl>
              <table>
                <thead>
                  <tr>
                    <th className="num">Rank</th>
                    <th>Similarity</th>
                    <th>Chunk</th>
                    <th>Why retrieved</th>
                  </tr>
                </thead>
                <tbody>
                  {run.results.map((r) => (
                    <tr key={r.chunk_id}>
                      <td className="num">{r.rank}</td>
                      <td>
                        <SimilarityBar value={r.similarity} />
                      </td>
                      <td>
                        <Link className="mono" style={{ fontSize: 11 }} to={`/chunks/${r.chunk_id}`}>
                          {r.chunk_id}
                        </Link>
                      </td>
                      <td style={{ fontSize: 11, color: "var(--text-dim)" }}>
                        {r.match_reasons.slice(0, 2).join(" · ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          ))}
        </Card>
      )}

      {data.model_runs.length > 0 && (
        <Card title="Model calls — what did the model see?">
          {data.model_runs.map((run) => (
            <Panel
              key={run.id}
              summary={
                <>
                  <strong>{run.agent_key}</strong>
                  <span className="badge neutral">{run.model}</span>
                  <span className="badge neutral">
                    {run.prompt_name}-{run.prompt_version}
                  </span>
                  <span style={{ flex: 1 }} />
                  <span className="mono" style={{ fontSize: 11, color: "var(--text-faint)" }}>
                    temp {run.temperature} · in {run.input_tokens} / out {run.output_tokens} ·{" "}
                    {ms(run.total_duration_ms)}
                    {run.tokens_per_second ? ` · ${run.tokens_per_second.toFixed(1)} tok/s` : ""}
                  </span>
                </>
              }
            >
              <Panel summary={<span>System prompt</span>}>
                <pre>{run.system_prompt}</pre>
              </Panel>
              <Panel summary={<span>User prompt (includes retrieved context)</span>}>
                <pre>{run.user_prompt}</pre>
              </Panel>
              <Panel summary={<span>Raw response</span>}>
                <pre>{run.raw_response}</pre>
              </Panel>
            </Panel>
          ))}
        </Card>
      )}

      {data.agent_runs.length > 0 && (
        <Card title="Agent runs — context isolation">
          <table>
            <thead>
              <tr>
                <th>Agent</th>
                <th>Status</th>
                <th>Prompt</th>
                <th className="num">Included</th>
                <th className="num">Excluded</th>
                <th className="num">Duration</th>
              </tr>
            </thead>
            <tbody>
              {data.agent_runs.map((run) => (
                <tr key={run.id}>
                  <td>{run.label}</td>
                  <td>
                    <StatusBadge status={run.status} />
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>
                    {run.prompt_name}-{run.prompt_version}
                  </td>
                  <td className="num">{run.included_chunk_ids?.length ?? 0}</td>
                  <td className="num" style={{ color: "var(--warn)" }}>
                    {run.excluded_chunk_ids?.length ?? 0}
                  </td>
                  <td className="num">{ms(run.duration_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </>
  );
}

function StageEntry({ stage }: { stage: TraceStage }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <div className="stage-row" onClick={() => setOpen(!open)}>
        <span className="ord">{String(stage.ordinal).padStart(2, "0")}</span>
        <span className="row" style={{ gap: 8 }}>
          <span className={`stage-dot ${stage.status}`} />
          {stage.label}
        </span>
        <span className="badge neutral">
          {stage.service} · {stage.transport}
        </span>
        <span className="num">{ms(stage.duration_ms)}</span>
      </div>
      {open && (
        <div style={{ padding: "10px 14px 14px 52px", background: "var(--bg)" }}>
          {stage.error && <ErrorNote error={stage.error} />}
          <div className="grid cols-2">
            <div>
              <strong style={{ fontSize: 12 }}>Input</strong>
              <Json value={stage.input} maxHeight={260} />
            </div>
            <div>
              <strong style={{ fontSize: 12 }}>Output</strong>
              <Json value={stage.output} maxHeight={260} />
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/** Visual node pipeline. Parallel agent stages are drawn side by side. */
function NodeGraph({ stages }: { stages: TraceStage[] }) {
  const parallelKeys = ["LEGAL_RESEARCH_AGENT", "SIMILAR_CASE_AGENT"];
  const parallel = stages.filter((s) => parallelKeys.includes(s.stage_key));
  const rest = stages.filter((s) => !parallelKeys.includes(s.stage_key));
  const before = rest.filter((s) => s.ordinal < (parallel[0]?.ordinal ?? Infinity));
  const after = rest.filter((s) => s.ordinal > (parallel[0]?.ordinal ?? -1));

  const Node = ({ stage }: { stage: TraceStage }) => (
    <div className={`node ${stage.status === "running" ? "active" : ""}`} title={stage.stage_key}>
      <div className="node-title">{stage.label}</div>
      <div className="node-meta">
        {stage.service} · {ms(stage.duration_ms)}
      </div>
      <div style={{ marginTop: 4 }}>
        <StatusBadge status={stage.status} />
      </div>
    </div>
  );

  return (
    <div className="node-graph">
      {before.map((stage) => (
        <div key={stage.id} style={{ display: "contents" }}>
          <Node stage={stage} />
          <div className="node-arrow" />
        </div>
      ))}
      {parallel.length > 0 && (
        <>
          <div className="node-parallel">
            {parallel.map((stage) => (
              <div key={stage.id} style={{ display: "flex", flexDirection: "column" }}>
                <Node stage={stage} />
              </div>
            ))}
          </div>
          <div className="node-arrow" />
          <div style={{ fontSize: 11, color: "var(--text-faint)", padding: "2px 0" }}>
            ↑ ran concurrently
          </div>
          <div className="node-arrow" />
        </>
      )}
      {after.map((stage, index) => (
        <div key={stage.id} style={{ display: "contents" }}>
          <Node stage={stage} />
          {index < after.length - 1 && <div className="node-arrow" />}
        </div>
      ))}
    </div>
  );
}
