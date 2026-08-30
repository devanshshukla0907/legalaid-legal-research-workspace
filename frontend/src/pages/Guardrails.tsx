/**
 * Guardrails: what the system refuses, and the evidence that it does.
 *
 * The page exists because a guardrail nobody can see is indistinguishable from one that is not
 * there. It runs the real adversarial suite against the live stack and reports both numbers that
 * matter: the catch rate on adversarial input, and the false-positive rate on legitimate legal
 * work. The second is the harder one - several probe cases deliberately contain words the
 * denylist screens for ("draft", "write", "ignore", "treatment"), because a legal tool that
 * refuses "draft a written statement in reply to the notice" is worse than one with no
 * guardrails at all.
 */

import { useState } from "react";
import { api, type GuardrailProbeCase, type GuardrailProbeResult } from "../api";
import { Card, ErrorNote, Metric, Panel, Spinner, useAsync } from "../components";

const LAYER_LABEL: Record<string, string> = {
  empty: "Empty",
  too_long: "Too long",
  injection: "Injection",
  denylist: "Denylist",
  corpus_distance: "Corpus distance",
  llm_scope: "LLM scope",
};

const KIND_TONE: Record<string, string> = {
  deterministic: "ok",
  embedding: "warn",
  model: "",
};

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function CaseRow({ probe }: { probe: GuardrailProbeCase }) {
  const wanted = probe.expected_blocked_by;
  const isLegit = wanted === null;
  return (
    <tr>
      <td>
        <code>{probe.id}</code>
      </td>
      <td style={{ maxWidth: 300 }}>
        <div>{probe.text}</div>
        {probe.note && (
          <small style={{ color: "var(--text-faint)" }}>{probe.note}</small>
        )}
      </td>
      <td>
        <span className={`badge ${isLegit ? "ok" : ""}`}>
          {isLegit ? "must allow" : `expect ${LAYER_LABEL[wanted] ?? wanted}`}
        </span>
      </td>
      <td>
        {probe.allowed ? (
          <span className="badge ok">allowed</span>
        ) : (
          <span className="badge err">
            blocked · {LAYER_LABEL[probe.blocked_by ?? ""] ?? probe.blocked_by}
          </span>
        )}
      </td>
      <td>
        {probe.correct ? (
          <span className="badge ok">correct</span>
        ) : (
          <span className="badge err">
            {isLegit ? "false positive" : "missed"}
          </span>
        )}
      </td>
      <td style={{ textAlign: "right", color: "var(--text-faint)" }}>
        {probe.latency_ms.toFixed(0)} ms
      </td>
    </tr>
  );
}

export default function Guardrails() {
  const config = useAsync(() => api.guardrailConfig(), []);
  const [result, setResult] = useState<GuardrailProbeResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [withCorpus, setWithCorpus] = useState(true);
  const [filter, setFilter] = useState<"all" | "failures" | "legitimate" | "adversarial">(
    "all",
  );

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.guardrailProbe(withCorpus));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const cases = (result?.cases ?? []).filter((probe) => {
    if (filter === "failures") return !probe.correct;
    if (filter === "legitimate") return probe.expected_blocked_by === null;
    if (filter === "adversarial") return probe.expected_blocked_by !== null;
    return true;
  });

  return (
    <>
      <div className="page-head">
        <h1>Guardrails</h1>
        <p>
          What the system accepts, what it refuses, and why. Deterministic checks run before the
          model, so an out-of-scope request never costs a generation.
        </p>
      </div>

      {config.error && <ErrorNote error={config.error} />}

      {config.data && (
        <Card title="The stack" actions={
          <span className={`badge ${config.data.enabled ? "ok" : "err"}`}>
            {config.data.enabled ? "enforced" : "disabled"}
          </span>
        }>
          <table className="table">
            <thead>
              <tr>
                <th>Layer</th>
                <th>Kind</th>
                <th>What it checks</th>
              </tr>
            </thead>
            <tbody>
              {config.data.layers.map((layer, index) => (
                <tr key={layer.layer}>
                  <td>
                    <code>{index + 1}. {LAYER_LABEL[layer.layer] ?? layer.layer}</code>
                  </td>
                  <td>
                    <span className={`badge ${KIND_TONE[layer.kind] ?? ""}`}>{layer.kind}</span>
                  </td>
                  <td>{layer.checks}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="row" style={{ gap: 24, marginTop: 14, flexWrap: "wrap" }}>
            <Metric
              label="Corpus similarity floor"
              value={config.data.corpus_similarity_floor.toFixed(2)}
            />
            <Metric label="Rate limit" value={config.data.rate_limit} />
            <Metric
              label="Auth"
              value={config.data.auth_enabled ? "required" : "open"}
            />
            <Metric
              label="GPU concurrency"
              value={String(config.data.llm_max_concurrency)}
            />
          </div>

          <Panel summary="How the similarity floor was chosen — measured, not guessed">
            <p style={{ margin: 0 }}>{config.data.floor_basis}</p>
          </Panel>

          <Panel summary="Output guardrail — applied after synthesis">
            <p style={{ margin: 0 }}>{config.data.output_guardrail}</p>
          </Panel>
        </Card>
      )}

      <Card
        title="Adversarial probe suite"
        actions={
          <div className="row" style={{ gap: 8 }}>
            <label className="row" style={{ gap: 6, fontSize: 12 }}>
              <input
                type="checkbox"
                checked={withCorpus}
                onChange={(event) => setWithCorpus(event.target.checked)}
              />
              embedding layer
            </label>
            <button onClick={run} disabled={busy}>
              {busy ? "Running..." : "Run suite"}
            </button>
          </div>
        }
      >
        <p style={{ marginTop: 0, color: "var(--text-dim)" }}>
          Every case is put through the live guardrail stack. Turning off the embedding layer
          shows how much the deterministic patterns catch on their own — the difference is what
          corpus distance is actually buying.
        </p>

        {error && <ErrorNote error={error} />}
        {busy && <Spinner label="screening every probe case" />}

        {result && (
          <>
            <div className="row" style={{ gap: 24, flexWrap: "wrap", marginBottom: 14 }}>
              <Metric
                label="Catch rate"
                value={pct(result.summary.catch_rate)}
                sub={`${result.summary.caught}/${result.summary.adversarial} adversarial`}
              />
              <Metric
                label="False positives"
                value={pct(result.summary.false_positive_rate)}
                sub={`${result.summary.false_positives}/${result.summary.legitimate} legitimate`}
              />
              <Metric label="Overall accuracy" value={pct(result.summary.accuracy)} />
              <Metric
                label="Caught by intended layer"
                value={`${result.summary.caught_by_expected_layer}/${result.summary.adversarial}`}
              />
              <Metric
                label="Mean screen cost"
                value={`${result.summary.mean_latency_ms.toFixed(0)} ms`}
                sub={`max ${result.summary.max_latency_ms.toFixed(0)} ms`}
              />
            </div>

            <Panel
              summary={`Which layer refused what — ${
                Object.keys(result.summary.blocked_by_layer).length
              } layers fired`}
            >
              <table className="table">
                <thead>
                  <tr>
                    <th>Layer</th>
                    <th style={{ textAlign: "right" }}>Refusals</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.summary.blocked_by_layer).map(([layer, count]) => (
                    <tr key={layer}>
                      <td>{LAYER_LABEL[layer] ?? layer}</td>
                      <td style={{ textAlign: "right" }}>{count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>

            <div className="row" style={{ gap: 6, margin: "14px 0 8px" }}>
              {(["all", "failures", "adversarial", "legitimate"] as const).map((key) => (
                <button
                  key={key}
                  className={filter === key ? "" : "ghost"}
                  onClick={() => setFilter(key)}
                >
                  {key}
                </button>
              ))}
            </div>

            <table className="table">
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Input</th>
                  <th>Expected</th>
                  <th>Actual</th>
                  <th>Verdict</th>
                  <th style={{ textAlign: "right" }}>Cost</th>
                </tr>
              </thead>
              <tbody>
                {cases.map((probe) => (
                  <CaseRow key={probe.id} probe={probe} />
                ))}
              </tbody>
            </table>
          </>
        )}
      </Card>
    </>
  );
}
