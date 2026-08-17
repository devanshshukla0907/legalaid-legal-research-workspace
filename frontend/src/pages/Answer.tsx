/**
 * Answer — the lawyer-facing result for a completed trace.
 *
 * Reads the recorded trace rather than re-running anything, so past research can be reopened
 * from the dashboard. Shows the synthesised answer first, then the supporting authorities, with
 * a single link into the developer view for anyone who wants the internals.
 */

import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { AnswerView } from "../AnswerView";
import { Card, Empty, ErrorNote, Spinner, useAsync } from "../components";

export default function Answer() {
  const { traceId } = useParams();
  const { data, error, loading } = useAsync(
    () => api.trace(traceId!),
    [traceId],
  );

  if (loading) return <Card><Spinner label="Loading your research…" /></Card>;
  if (error) return <ErrorNote error={error} />;
  if (!data) return null;

  const recorded = (data.answer ?? {}) as Record<string, any>;
  const answer = recorded.synthesis ?? {};
  const legal = recorded.legal ?? {};
  const cases = recorded.cases ?? {};
  const draft = recorded.draft ?? {};

  return (
    <>
      <div className="page-head">
        <h1>Research result</h1>
        <p style={{ fontSize: 15, color: "var(--text)" }}>{data.question}</p>
        <div className="row" style={{ marginTop: 8, gap: 8 }}>
          <span style={{ fontSize: 12, color: "var(--text-faint)" }}>
            {data.started_at ? new Date(data.started_at).toLocaleString() : ""}
          </span>
          <Link className="btn small" to={`/pipeline/${data.trace_id}`}>
            How was this produced?
          </Link>
        </div>
      </div>

      {Object.keys(answer).length === 0 ? (
        <Empty>
          This research ran before the answer stage existed, or it did not complete. Open the
          pipeline trace to see what happened.
        </Empty>
      ) : (
        <AnswerView answer={answer} legal={legal} cases={cases} draft={draft} />
      )}
    </>
  );
}
