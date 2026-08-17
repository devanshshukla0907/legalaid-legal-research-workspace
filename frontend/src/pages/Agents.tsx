/** Agents: the roster, each agent's responsibility, prompt version and context policy. */

import { Link } from "react-router-dom";
import { api, type AgentDescriptor } from "../api";
import { Card, Empty, ErrorNote, Panel, Spinner, StatusBadge, ms, useAsync } from "../components";

export default function Agents() {
  const { data, error, loading } = useAsync(() => api.agents(), []);
  const prompts = useAsync(() => api.prompts(), []);

  if (loading) return <Card><Spinner label="Loading agents…" /></Card>;
  if (error) return <ErrorNote error={error} />;

  return (
    <>
      <div className="page-head">
        <h1>Agents</h1>
        <p>
          Six specialised roles. Each has its own prompt, its own model parameters, and its own
          context policy — the “withheld” column is what makes context isolation verifiable.
        </p>
      </div>

      <Card title="Orchestration order">
        <div className="node-graph">
          <div className="node">
            <div className="node-title">Intake Router</div>
            <div className="node-meta">classify + rewrite query</div>
          </div>
          <div className="node-arrow" />
          <div className="node">
            <div className="node-title">Retrieval (hybrid)</div>
            <div className="node-meta">dense + BM25 + graph</div>
          </div>
          <div className="node-arrow" />
          <div className="node">
            <div className="node-title">Document Understanding</div>
            <div className="node-meta">only when documents exist</div>
          </div>
          <div className="node-arrow" />
          <div className="node-parallel">
            <div className="node">
              <div className="node-title">Legal Research</div>
              <div className="node-meta">statute only</div>
            </div>
            <div className="node">
              <div className="node-title">Similar Cases</div>
              <div className="node-meta">precedent only</div>
            </div>
          </div>
          <div className="node-arrow" />
          <div style={{ fontSize: 11, color: "var(--text-faint)" }}>↑ concurrent</div>
          <div className="node-arrow" />
          <div className="node">
            <div className="node-title">Verification (critic)</div>
            <div className="node-meta">audit → retry loop</div>
          </div>
          <div className="node-arrow" />
          <div className="node">
            <div className="node-title">Drafting</div>
            <div className="node-meta">findings only, no raw chunks</div>
          </div>
        </div>
      </Card>

      {(data?.agents ?? []).map((agent) => (
        <AgentCard
          key={agent.key}
          agent={agent}
          prompt={prompts.data?.prompts?.find((p: any) => p.name === agent.prompt_name)}
        />
      ))}
    </>
  );
}

function AgentCard({ agent, prompt }: { agent: AgentDescriptor; prompt?: any }) {
  return (
    <Card
      title={
        <span className="row" style={{ gap: 8 }}>
          <span className="badge neutral">{agent.order}</span>
          {agent.label}
        </span>
      }
      actions={
        <span className="row" style={{ gap: 7 }}>
          {agent.prompt_version && (
            <span className="badge law">
              {agent.prompt_name}-{agent.prompt_version}
            </span>
          )}
          {agent.temperature != null && (
            <span className="badge neutral">temp {agent.temperature}</span>
          )}
          {agent.max_tokens != null && (
            <span className="badge neutral">max {agent.max_tokens} tok</span>
          )}
        </span>
      }
    >
      <dl className="kv">
        <dt>Responsibility</dt>
        <dd>{agent.responsibility}</dd>
        <dt>Context it sees</dt>
        <dd style={{ color: "var(--ok)" }}>{agent.sees}</dd>
        <dt>Deliberately withheld</dt>
        <dd style={{ color: "var(--warn)" }}>{agent.withheld}</dd>
        <dt>Runs recorded</dt>
        <dd className="mono">
          {agent.runs}
          {agent.avg_latency_ms != null && ` · avg ${ms(agent.avg_latency_ms)}`}
          {agent.success_rate != null &&
            ` · ${Math.round(agent.success_rate * 100)}% success`}
        </dd>
      </dl>

      {agent.recent.length > 0 && (
        <Panel summary={<span>Recent executions ({agent.recent.length})</span>}>
          <table>
            <thead>
              <tr>
                <th>Trace</th>
                <th>Status</th>
                <th className="num">Duration</th>
                <th className="num">Included</th>
                <th className="num">Excluded</th>
              </tr>
            </thead>
            <tbody>
              {agent.recent.map((run) => (
                <tr key={run.id}>
                  <td>
                    <Link className="mono" style={{ fontSize: 11 }} to={`/pipeline/${run.trace_id}`}>
                      {run.trace_id}
                    </Link>
                  </td>
                  <td><StatusBadge status={run.status} /></td>
                  <td className="num">{ms(run.duration_ms)}</td>
                  <td className="num">{run.included}</td>
                  <td className="num" style={{ color: "var(--warn)" }}>{run.excluded}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}

      {prompt && (
        <>
          <Panel summary={<span>System prompt (version {prompt.version})</span>}>
            <pre>{prompt.system_prompt}</pre>
          </Panel>
          <Panel summary={<span>User prompt template</span>}>
            <pre>{prompt.user_prompt}</pre>
          </Panel>
        </>
      )}
      {!prompt && agent.prompt_name && <Empty>Prompt file not loaded.</Empty>}
    </Card>
  );
}
