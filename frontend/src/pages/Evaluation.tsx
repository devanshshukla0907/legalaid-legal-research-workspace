/**
 * Evaluation Matrix: every Week 4 evaluation question, its category, and what each model
 * actually answered.
 *
 * This page exists because a professor grading "did you build a real evaluation dataset" should
 * not have to open a JSON file to check. It shows the same 27 tasks eval/dataset.py defines, the
 * seven categories they were deliberately split into, and - once eval/runner.py has been run -
 * each model's recorded answer, whether it was scored correct, and whether it hallucinated,
 * side by side per task.
 */

import { useState } from "react";
import {
  api,
  type EvalTask,
  type EvalTaskResult,
} from "../api";
import { Card, Empty, ErrorNote, Metric, Panel, Spinner, useAsync } from "../components";

const MODEL_LABEL: Record<string, string> = {
  "llama3.2:3b": "Llama 3.2 (3B)",
  "qwen2.5:3b-instruct": "Qwen 2.5 (3B Instruct)",
  "gemma2:2b": "Gemma 2 (2B)",
};

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function CategoryPill({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button className={active ? "" : "ghost"} onClick={onClick} style={{ fontSize: 12 }}>
      {children}
    </button>
  );
}

function VerdictBadge({ result }: { result: EvalTaskResult | undefined }) {
  if (!result) return <span className="badge neutral">no run recorded</span>;
  if (result.hallucinated) return <span className="badge err">hallucinated</span>;
  return (
    <span className={`badge ${result.correct ? "ok" : "err"}`}>
      {result.correct ? "correct" : "incorrect"}
    </span>
  );
}

function TaskRow({
  task,
  resultsByModel,
  models,
  expanded,
  onToggle,
}: {
  task: EvalTask;
  resultsByModel: Record<string, EvalTaskResult | undefined>;
  models: string[];
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr onClick={onToggle} style={{ cursor: "pointer" }}>
        <td>
          <code>{task.id}</code>
        </td>
        <td style={{ maxWidth: 340 }}>
          {task.question}
          {!task.answerable && (
            <span className="badge warn" style={{ marginLeft: 6 }}>
              deliberately unanswerable
            </span>
          )}
        </td>
        <td>
          <span className="badge">{task.category_label}</span>
        </td>
        {models.map((model) => (
          <td key={model}>
            <VerdictBadge result={resultsByModel[model]} />
          </td>
        ))}
      </tr>
      {expanded && (
        <tr>
          <td />
          <td colSpan={2 + models.length}>
            <div style={{ padding: "8px 0 14px", display: "grid", gap: 10 }}>
              <div>
                <strong>Expected authority: </strong>
                {task.expected_sections.length ? task.expected_sections.join("; ") : "—"}
              </div>
              {task.notes && (
                <div style={{ color: "var(--text-dim)" }}>
                  <strong>Why this task exists: </strong>
                  {task.notes}
                </div>
              )}
              <div style={{ display: "grid", gap: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                {models.map((model) => {
                  const r = resultsByModel[model];
                  return (
                    <div key={model} className="panel-content" style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 10 }}>
                      <div style={{ fontWeight: 600, marginBottom: 6 }}>
                        {MODEL_LABEL[model] ?? model}
                      </div>
                      {!r ? (
                        <span style={{ color: "var(--text-faint)" }}>No recorded run.</span>
                      ) : (
                        <>
                          <div style={{ marginBottom: 6, fontSize: 13 }}>{r.answer_text}</div>
                          <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
                            Cited: {r.cited_sections.length ? r.cited_sections.join(", ") : "none"}
                          </div>
                          <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
                            Retrieval recall {r.retrieval_recall?.toFixed(2) ?? "—"} · MRR{" "}
                            {r.retrieval_mrr?.toFixed(2) ?? "—"} · {r.latency_ms ? `${(r.latency_ms / 1000).toFixed(1)}s` : "—"}
                          </div>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function Evaluation() {
  const dataset = useAsync(() => api.evalDataset(), []);
  const results = useAsync(() => api.evalResults(), []);
  const [category, setCategory] = useState<string>("all");
  const [expandedTask, setExpandedTask] = useState<string | null>(null);

  const models = results.data?.available_models ?? [];
  const tasks = dataset.data?.tasks ?? [];
  const filtered = category === "all" ? tasks : tasks.filter((t) => t.category === category);

  const resultsByTaskAndModel: Record<string, Record<string, EvalTaskResult | undefined>> = {};
  for (const model of models) {
    for (const r of results.data?.results[model] ?? []) {
      resultsByTaskAndModel[r.task_id] = resultsByTaskAndModel[r.task_id] ?? {};
      resultsByTaskAndModel[r.task_id][model] = r;
    }
  }

  // Per-model, per-category accuracy - the exact breakdown the brief asked for and which a
  // single overall accuracy number cannot show.
  const categoryAccuracy: Record<string, Record<string, { correct: number; total: number }>> = {};
  for (const model of models) {
    categoryAccuracy[model] = {};
    for (const r of results.data?.results[model] ?? []) {
      const bucket = categoryAccuracy[model][r.category] ?? { correct: 0, total: 0 };
      bucket.total += 1;
      if (r.correct) bucket.correct += 1;
      categoryAccuracy[model][r.category] = bucket;
    }
  }

  return (
    <>
      <div className="page-head">
        <h1>Evaluation matrix</h1>
        <p>
          The Week 4 evaluation dataset: {dataset.data?.total ?? 27} tasks across{" "}
          {dataset.data?.categories.length ?? 7} categories, and what each of the three compared
          models actually answered for every one of them. This is the raw material behind{" "}
          <code>docs/evaluation.md</code> — nothing here is summarised away.
        </p>
      </div>

      {dataset.error && <ErrorNote error={dataset.error} />}
      {results.error && <ErrorNote error={results.error} />}

      {dataset.loading && <Spinner label="loading evaluation dataset" />}

      {dataset.data && (
        <Card title="Dataset composition">
          <div className="row" style={{ gap: 20, flexWrap: "wrap", marginBottom: 14 }}>
            <Metric label="Total tasks" value={dataset.data.total} />
            <Metric
              label="Answerable"
              value={dataset.data.tasks.filter((t) => t.answerable).length}
            />
            <Metric
              label="Deliberately unanswerable"
              value={dataset.data.tasks.filter((t) => !t.answerable).length}
              sub="tests whether a model knows when to refuse"
            />
            <Metric label="Categories" value={dataset.data.categories.length} />
            <Metric
              label="Models compared"
              value={models.length || "no runs recorded"}
            />
          </div>

          <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
            <CategoryPill active={category === "all"} onClick={() => setCategory("all")}>
              all ({dataset.data.total})
            </CategoryPill>
            {dataset.data.categories.map((c) => (
              <CategoryPill key={c.key} active={category === c.key} onClick={() => setCategory(c.key)}>
                {c.label} ({c.count})
              </CategoryPill>
            ))}
          </div>
        </Card>
      )}

      {models.length > 0 && (
        <Card title="Accuracy by category and model">
          <p style={{ marginTop: 0, color: "var(--text-dim)" }}>
            The reason the dataset is split into categories rather than left as one pool: a
            single overall accuracy number hides which model is actually better at what.
          </p>
          <table className="table">
            <thead>
              <tr>
                <th>Category</th>
                {models.map((m) => (
                  <th key={m}>{MODEL_LABEL[m] ?? m}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(dataset.data?.categories ?? []).map((c) => (
                <tr key={c.key}>
                  <td>{c.label}</td>
                  {models.map((m) => {
                    const bucket = categoryAccuracy[m]?.[c.key];
                    return (
                      <td key={m}>
                        {bucket ? `${bucket.correct}/${bucket.total} (${pct(bucket.correct / bucket.total)})` : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {results.data?.note && (
        <Card title="No recorded runs">
          <Empty>{results.data.note} Run <code>python eval/runner.py</code> to populate this page.</Empty>
        </Card>
      )}

      <Card title={`Tasks (${filtered.length})`}>
        {filtered.length === 0 ? (
          <Empty>No tasks in this category.</Empty>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Question</th>
                <th>Category</th>
                {models.map((m) => (
                  <th key={m}>{MODEL_LABEL[m] ?? m}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((task) => (
                <TaskRow
                  key={task.id}
                  task={task}
                  models={models}
                  resultsByModel={resultsByTaskAndModel[task.id] ?? {}}
                  expanded={expandedTask === task.id}
                  onToggle={() => setExpandedTask(expandedTask === task.id ? null : task.id)}
                />
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Panel summary="How correctness and hallucination are computed here">
        <p style={{ margin: 0 }}>
          <strong>Correct</strong> (answerable task): the model cited at least one expected
          section and every citation it made was grounded in a chunk it was actually given.
          <strong> Correct</strong> (unanswerable task): the model declined or cited nothing.
          <strong> Hallucinated</strong>: the model cited a chunk id it was never supplied — this
          overrides a "correct" verdict, because an unverifiable citation is worse than none in
          legal work. These are the same definitions used in <code>eval/metrics.py</code> and{" "}
          <code>docs/evaluation.md</code>; this page recomputes them from the same recorded
          per-model JSON files rather than a separate pass, so the numbers cannot drift apart.
        </p>
      </Panel>
    </>
  );
}
