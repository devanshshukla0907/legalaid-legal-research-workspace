/**
 * Dashboard — the lawyer's view.
 *
 * Deliberately not a metrics console. A practising advocate cares about what the library covers,
 * what work has been done, and how to start the next piece of research. Vector counts, token
 * throughput and trace internals live in the Developer section instead.
 */

import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, type Health } from "../api";
import { Card, Empty, Spinner, useAsync } from "../components";

const STARTERS = [
  {
    title: "Breach of contract",
    prompt:
      "My client supplied goods under a written agreement and the buyer refused payment after delivery. What provisions apply to claiming compensation, and are there similar Indian cases?",
  },
  {
    title: "Penalty clause",
    prompt:
      "The agreement stipulates a fixed sum payable on default. Is that penalty clause enforceable, and how much can realistically be recovered?",
  },
  {
    title: "Electronic evidence",
    prompt:
      "The agreement and all correspondence exist only as email and PDF. How do I prove these electronic records in court?",
  },
  {
    title: "Reply to a notice",
    prompt:
      "Draft a reply to a legal notice alleging breach of a supply agreement, denying liability and asserting that payment was withheld for defective goods.",
  },
];

export default function Dashboard({ health }: { health: Health | null }) {
  const navigate = useNavigate();
  const traces = useAsync(() => api.traces(6), []);
  const knowledge = useAsync(() => api.knowledge(), []);
  const [question, setQuestion] = useState("");

  function start(prompt: string) {
    // Hand the question to the Research page rather than duplicating the pipeline UI here.
    sessionStorage.setItem("legalaid-pending-question", prompt);
    navigate("/research");
  }

  if (!health) {
    return (
      <Card>
        <Spinner label="Connecting to LegalAId…" />
      </Card>
    );
  }

  const acts = knowledge.data?.acts ?? [];
  const cases = knowledge.data?.cases ?? [];

  return (
    <>
      <div className="hero">
        <h1>Good day, counsel.</h1>
        <p>
          Ask a question, paste an FIR or notice, or upload your matter documents. LegalAId
          searches Indian legislation and a curated set of Supreme Court judgments, then reports
          back with the applicable provisions, comparable precedent and its reasoning.
        </p>

        <div className="hero-ask">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Describe the matter, or paste the document text…"
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && question.trim()) {
                start(question);
              }
            }}
          />
          <div className="row between" style={{ marginTop: 10 }}>
            <span style={{ fontSize: 12, color: "var(--text-faint)" }}>
              Ctrl / ⌘ + Enter to begin
            </span>
            <button
              className="primary"
              disabled={!question.trim()}
              onClick={() => start(question)}
            >
              Start research
            </button>
          </div>
        </div>
      </div>

      <div className="section-label">Common starting points</div>
      <div className="grid cols-4">
        {STARTERS.map((starter) => (
          <button key={starter.title} className="starter" onClick={() => start(starter.prompt)}>
            <strong>{starter.title}</strong>
            <span>{starter.prompt.slice(0, 96)}…</span>
          </button>
        ))}
      </div>

      <div className="split" style={{ marginTop: 18 }}>
        <div>
          <div className="section-label">Your recent research</div>
          <Card tight>
            {!traces.data?.traces?.length ? (
              <Empty>
                Nothing yet. Ask a question above and it will appear here.
              </Empty>
            ) : (
              <table>
                <tbody>
                  {traces.data.traces.map((trace: any) => (
                    <tr
                      key={trace.trace_id}
                      className="clickable"
                      onClick={() => navigate(`/answer/${trace.trace_id}`)}
                    >
                      <td>
                        <div style={{ fontSize: 13, lineHeight: 1.45 }}>
                          {trace.question?.slice(0, 130) || "(no question recorded)"}
                          {trace.question?.length > 130 ? "…" : ""}
                        </div>
                        <div style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 3 }}>
                          {trace.started_at
                            ? new Date(trace.started_at).toLocaleString()
                            : "—"}
                          {trace.status !== "success" && ` · ${trace.status}`}
                        </div>
                      </td>
                      <td style={{ width: 90, textAlign: "right" }}>
                        <span className="link-ish">Open →</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
        </div>

        <div>
          <div className="section-label">What LegalAId can search</div>
          <Card tight>
            {knowledge.loading ? (
              <div style={{ padding: 14 }}>
                <Spinner />
              </div>
            ) : (
              <>
                <table>
                  <tbody>
                    {acts.map((act: any) => (
                      <tr key={act.act_name}>
                        <td>
                          <div style={{ fontSize: 13 }}>{act.act_name}</div>
                          <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
                            {act.provisions} sections · India Code (official)
                          </div>
                        </td>
                      </tr>
                    ))}
                    <tr>
                      <td>
                        <div style={{ fontSize: 13 }}>Supreme Court judgments</div>
                        <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
                          {cases.length} curated reports · e-SCR (CC-BY-4.0)
                        </div>
                      </td>
                    </tr>
                  </tbody>
                </table>
                <div style={{ padding: "10px 12px" }}>
                  <Link className="btn small" to="/knowledge">
                    Browse the library
                  </Link>
                </div>
              </>
            )}
          </Card>

          <div className="notice" style={{ marginTop: 12 }}>
            <strong>Everything is verifiable.</strong> Every provision and judgment links back to
            its official source, and the Developer section shows exactly how each answer was
            produced.
          </div>
        </div>
      </div>

      <div className="disclaimer" style={{ marginTop: 18 }}>
        <strong>AI-generated legal research for professional review.</strong> Verify statutes,
        amendments, case status, citations and procedural requirements before relying on this
        output. LegalAId does not provide legal advice.
      </div>
    </>
  );
}
