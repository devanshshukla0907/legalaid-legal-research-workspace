/**
 * The lawyer-facing answer.
 *
 * Order matters here: the prose answer comes first because that is what was actually asked for.
 * Retrieved authorities follow as support. Crucially, the system's own reasoning is labelled
 * separately from what the sources say, so a reader can always tell inference from law.
 */

import { Link } from "react-router-dom";
import { Card, Empty } from "./components";

/** Minimal markdown rendering — paragraphs, bold, and simple bullets. */
function Prose({ text }: { text: string }) {
  const blocks = text.trim().split(/\n\s*\n/);
  return (
    <div className="prose">
      {blocks.map((block, index) => {
        const lines = block.split("\n");
        const isList = lines.every((l) => /^\s*[-*•]\s+/.test(l));
        if (isList) {
          return (
            <ul key={index}>
              {lines.map((line, i) => (
                <li key={i}>{inline(line.replace(/^\s*[-*•]\s+/, ""))}</li>
              ))}
            </ul>
          );
        }
        if (/^#{1,4}\s/.test(block)) {
          return <h4 key={index}>{inline(block.replace(/^#{1,4}\s*/, ""))}</h4>;
        }
        return <p key={index}>{inline(block)}</p>;
      })}
    </div>
  );
}

function inline(text: string) {
  // Bold spans only; anything richer is not worth a markdown dependency here.
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, index) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={index}>{part.slice(2, -2)}</strong>
    ) : (
      <span key={index}>{part}</span>
    ),
  );
}

export function AnswerView({
  answer,
  legal,
  cases,
  draft,
}: {
  answer: Record<string, any>;
  legal: Record<string, any>;
  cases: Record<string, any>;
  draft: Record<string, any>;
}) {
  const confidence = String(answer.confidence ?? "moderate");
  const sections = (legal?.sections ?? []) as any[];
  const precedents = (cases?.cases ?? []) as any[];

  return (
    <>
      <div className="answer-card">
        <div className="answer-head">
          <span className="answer-avatar">LA</span>
          <div style={{ flex: 1 }}>
            <strong>LegalAId</strong>
            <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
              Research assistant · for professional review
            </div>
          </div>
          <span
            className={`badge ${confidence === "high" ? "ok" : confidence === "low" ? "err" : "warn"}`}
            title="How squarely the retrieved material answers the question"
          >
            {confidence} confidence
          </span>
        </div>

        {answer.used_fallback && (
          <div className="disclaimer" style={{ margin: "0 0 12px" }}>
            The synthesis step did not complete, so this summary was assembled directly from the
            structured findings rather than written by the model.
          </div>
        )}

        <Prose text={String(answer.answer_markdown ?? "")} />

        {Array.isArray(answer.key_points) && answer.key_points.length > 0 && (
          <div className="answer-block">
            <div className="answer-block-label">Key points</div>
            <ul className="clean-list">
              {answer.key_points.map((point: string, index: number) => (
                <li key={index}>{point}</li>
              ))}
            </ul>
          </div>
        )}

        {Array.isArray(answer.inference) && answer.inference.length > 0 && (
          <div className="answer-block inference">
            <div className="answer-block-label">
              AI inference
              <span className="hint">
                the system's own reasoning, not a quotation from any source
              </span>
            </div>
            <ul className="clean-list">
              {answer.inference.map((item: string, index: number) => (
                <li key={index}>{item}</li>
              ))}
            </ul>
          </div>
        )}

        {Array.isArray(answer.uncertainties) && answer.uncertainties.length > 0 && (
          <div className="answer-block uncertain">
            <div className="answer-block-label">Uncertain / not established</div>
            <ul className="clean-list">
              {answer.uncertainties.map((item: string, index: number) => (
                <li key={index}>{item}</li>
              ))}
            </ul>
          </div>
        )}

        {Array.isArray(answer.next_steps) && answer.next_steps.length > 0 && (
          <div className="answer-block next">
            <div className="answer-block-label">Suggested next steps</div>
            <ol className="clean-list">
              {answer.next_steps.map((item: string, index: number) => (
                <li key={index}>{item}</li>
              ))}
            </ol>
          </div>
        )}
      </div>

      <div className="section-label">Supporting authorities</div>
      <div className="grid cols-2">
        <Card title={`Applicable provisions (${sections.length})`}>
          {sections.length === 0 ? (
            <Empty>
              {legal?.evidence_sufficient === false
                ? "The agent reported the statutory evidence as insufficient rather than guessing."
                : "No applicable provision was identified in the retrieved material."}
            </Empty>
          ) : (
            sections.map((section, index) => (
              <div key={index} className="authority">
                <div className="row" style={{ gap: 7 }}>
                  <span className="badge law">{section.section}</span>
                  <strong style={{ fontSize: 13 }}>{section.act}</strong>
                </div>
                <p style={{ margin: "5px 0 3px", fontSize: 13 }}>
                  {section.what_it_provides}
                </p>
                <div style={{ fontSize: 12, color: "var(--text-dim)" }}>
                  Why it matters: {section.relevance}
                </div>
                {section.chunk_id && (
                  <Link className="source-link" to={`/chunks/${section.chunk_id}`}>
                    View the exact text this came from →
                  </Link>
                )}
              </div>
            ))
          )}
        </Card>

        <Card title={`Comparable judgments (${precedents.length})`}>
          {precedents.length === 0 ? (
            <Empty>
              {cases?.note ??
                "No sufficiently similar judgment was found in the curated case corpus."}
            </Empty>
          ) : (
            precedents.map((precedent, index) => (
              <div key={index} className="authority">
                <div className="row" style={{ gap: 7 }}>
                  <span className="badge case">{precedent.citation}</span>
                  {precedent.strength && (
                    <span
                      className={`badge ${
                        precedent.strength === "strong"
                          ? "ok"
                          : precedent.strength === "weak"
                            ? "warn"
                            : "neutral"
                      }`}
                    >
                      {precedent.strength}
                    </span>
                  )}
                </div>
                <strong style={{ fontSize: 13, display: "block", marginTop: 4 }}>
                  {precedent.case_title}
                </strong>
                {precedent.relevant_principle && (
                  <p style={{ margin: "4px 0 3px", fontSize: 13 }}>
                    {precedent.relevant_principle}
                  </p>
                )}
                {Array.isArray(precedent.distinguishing_factors) &&
                  precedent.distinguishing_factors.length > 0 && (
                    <div style={{ fontSize: 12, color: "var(--warn)" }}>
                      Distinguishable: {precedent.distinguishing_factors.join("; ")}
                    </div>
                  )}
                {precedent.chunk_id && (
                  <Link className="source-link" to={`/chunks/${precedent.chunk_id}`}>
                    View the judgment extract →
                  </Link>
                )}
              </div>
            ))
          )}
        </Card>
      </div>

      {draft && draft.draft_markdown && (
        <>
          <div className="section-label">Draft prepared for you</div>
          <Card
            title={String(draft.title ?? "Draft")}
            actions={<span className="badge warn">first draft — review before use</span>}
          >
            <div className="draft-body">{String(draft.draft_markdown)}</div>
            <div className="grid cols-3" style={{ marginTop: 12 }}>
              <SmallList title="Authorities relied on" items={draft.authorities_relied_on} />
              <SmallList title="You must supply" items={draft.missing_information} warn />
              <SmallList title="Review points" items={draft.review_points} />
            </div>
          </Card>
        </>
      )}
    </>
  );
}

function SmallList({
  title,
  items,
  warn,
}: {
  title: string;
  items: unknown;
  warn?: boolean;
}) {
  const list = Array.isArray(items) ? (items as string[]) : [];
  return (
    <div>
      <div
        className="answer-block-label"
        style={warn ? { color: "var(--warn)" } : undefined}
      >
        {title}
      </div>
      {list.length === 0 ? (
        <div style={{ fontSize: 12, color: "var(--text-faint)" }}>—</div>
      ) : (
        <ul className="clean-list" style={{ fontSize: 12 }}>
          {list.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
