/** System: service health, models, configuration and exercise coverage. */

import { api, type Health } from "../api";
import { Card, Empty, Metric, Spinner, StatusBadge, useAsync } from "../components";

export default function System({ health }: { health: Health | null }) {
  const models = useAsync(() => api.models(), []);
  const prompts = useAsync(() => api.prompts(), []);

  if (!health) return <Card><Spinner label="Connecting…" /></Card>;

  return (
    <>
      <div className="page-head">
        <h1>System</h1>
        <p>Runtime state of the services, the models and the knowledge infrastructure.</p>
      </div>

      <div className="grid cols-4">
        <Metric
          label="API"
          value={<StatusBadge status={health.status} />}
          sub={`environment ${health.environment}`}
        />
        <Metric
          label="Ollama"
          value={<StatusBadge status={health.ollama.reachable ? "ok" : "failed"} />}
          sub={health.ollama.base_url}
        />
        <Metric label="Transport" value={health.transport} sub="inter-service calls" />
        <Metric label="Active phase" value={health.active_phase} sub="of 5 exercises" />
      </div>

      <Card title="Models">
        <dl className="kv">
          <dt>Generation (default)</dt>
          <dd className="mono">
            {health.ollama.llm_model}{" "}
            <StatusBadge status={health.ollama.llm_model_available ? "ok" : "failed"} />
          </dd>
          <dt>Embeddings</dt>
          <dd className="mono">
            {health.ollama.embedding_model}{" "}
            <StatusBadge status={health.ollama.embedding_model_available ? "ok" : "failed"} />
          </dd>
          <dt>Context window</dt>
          <dd className="mono">{models.data?.num_ctx ?? "—"} tokens</dd>
          <dt>Temperature (default)</dt>
          <dd className="mono">{models.data?.temperature ?? "—"}</dd>
        </dl>

        {models.data && (
          <table style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th>Installed model</th>
                <th>Family</th>
                <th>Parameters</th>
                <th>Quantisation</th>
                <th className="num">Size</th>
                <th>Usable for generation</th>
              </tr>
            </thead>
            <tbody>
              {models.data.installed.map((model) => (
                <tr key={model.name}>
                  <td className="mono" style={{ fontSize: 12 }}>{model.name}</td>
                  <td style={{ fontSize: 12 }}>{model.family ?? "—"}</td>
                  <td style={{ fontSize: 12 }}>{model.parameter_size ?? "—"}</td>
                  <td style={{ fontSize: 12 }}>{model.quantization ?? "—"}</td>
                  <td className="num">{(model.size_bytes / 1e9).toFixed(2)} GB</td>
                  <td>
                    <span className={`badge ${model.usable_for_generation ? "ok" : "neutral"}`}>
                      {model.usable_for_generation ? "yes" : "embedding only"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Card title="Retrieval infrastructure">
        <div className="grid cols-3">
          <div>
            <strong style={{ fontSize: 12 }}>Dense (FAISS)</strong>
            <dl className="kv" style={{ marginTop: 6 }}>
              <dt>Vectors</dt>
              <dd className="mono">{health.retrievers.dense.vectors}</dd>
              <dt>Dimensions</dt>
              <dd className="mono">{health.retrievers.dense.dimensions}</dd>
            </dl>
          </div>
          <div>
            <strong style={{ fontSize: 12 }}>Keyword (BM25)</strong>
            <dl className="kv" style={{ marginTop: 6 }}>
              <dt>Documents</dt>
              <dd className="mono">{health.retrievers.keyword.documents}</dd>
              <dt>Vocabulary</dt>
              <dd className="mono">{health.retrievers.keyword.vocabulary}</dd>
            </dl>
          </div>
          <div>
            <strong style={{ fontSize: 12 }}>Graph (symbolic)</strong>
            <dl className="kv" style={{ marginTop: 6 }}>
              <dt>Nodes</dt>
              <dd className="mono">{health.retrievers.graph.nodes}</dd>
              <dt>Edges</dt>
              <dd className="mono">{health.retrievers.graph.edges}</dd>
            </dl>
          </div>
        </div>
      </Card>

      <Card title="Data counts">
        <table>
          <tbody>
            {Object.entries(health.counts).map(([key, value]) => (
              <tr key={key}>
                <td style={{ fontSize: 13 }}>{key.replace(/_/g, " ")}</td>
                <td className="num">{value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Prompt registry">
        {!prompts.data?.prompts?.length ? (
          <Empty>No prompts found.</Empty>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Prompt</th>
                <th>Version</th>
                <th className="num">Temp</th>
                <th className="num">Max tokens</th>
                <th>Variables</th>
              </tr>
            </thead>
            <tbody>
              {prompts.data.prompts.map((prompt: any) => (
                <tr key={`${prompt.name}-${prompt.version}`}>
                  <td className="mono" style={{ fontSize: 12 }}>{prompt.name}</td>
                  <td>
                    <span className="badge law">{prompt.version}</span>
                  </td>
                  <td className="num">{prompt.temperature}</td>
                  <td className="num">{prompt.max_tokens}</td>
                  <td style={{ fontSize: 11, color: "var(--text-faint)" }}>
                    {prompt.variables.join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Card title="Capability matrix">
        <table>
          <thead>
            <tr>
              <th>Capability</th>
              <th className="num">Phase</th>
              <th>Exercise</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {health.capabilities.map((capability) => (
              <tr key={capability.key}>
                <td>
                  <strong style={{ fontSize: 13 }}>{capability.label}</strong>
                  <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
                    {capability.description}
                  </div>
                </td>
                <td className="num">{capability.phase}</td>
                <td style={{ fontSize: 11 }}>{capability.exercise}</td>
                <td>
                  <span className={`badge ${capability.enabled ? "ok" : "neutral"}`}>
                    {capability.enabled ? "active" : "inactive"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </>
  );
}
