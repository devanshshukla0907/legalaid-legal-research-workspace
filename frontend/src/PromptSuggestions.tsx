/**
 * Prompt suggestions: as the user types, suggest questions the corpus can actually answer.
 *
 * Debounced 250ms and matched against a static, corpus-grounded index (/api/suggest) rather
 * than an LLM call - the Intake Router could theoretically classify partial input, but it is a
 * multi-second generation, so re-running it per keystroke would either lag badly or spam the
 * GPU queue every time the user pauses. Prefix matching against real section headings and case
 * titles is near-instant and every suggestion is grounded in something retrievable, unlike a
 * hardcoded example list.
 */

import { useEffect, useRef, useState } from "react";
import { api } from "./api";

interface Suggestion {
  text: string;
  kind: string;
  detail: string;
}

export function PromptSuggestions({
  value,
  onPick,
  disabled,
}: {
  value: string;
  onPick: (text: string) => void;
  disabled?: boolean;
}) {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [open, setOpen] = useState(false);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (timer.current) window.clearTimeout(timer.current);
    // Suggest against the last word or two being typed, not the whole accumulated question -
    // "compensation for breach of" should still surface Section 73 suggestions.
    const tail = value.trim().split(/\s+/).slice(-3).join(" ");
    if (disabled || tail.length < 3) {
      setSuggestions([]);
      setOpen(false);
      return;
    }
    timer.current = window.setTimeout(async () => {
      try {
        const response = await api.suggest(tail, 5);
        setSuggestions(response.suggestions);
        setOpen(response.suggestions.length > 0);
      } catch {
        setSuggestions([]);
        setOpen(false);
      }
    }, 250);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [value, disabled]);

  if (!open || suggestions.length === 0) return null;

  return (
    <div
      className="card"
      style={{
        position: "absolute",
        top: "100%",
        left: 0,
        right: 0,
        zIndex: 20,
        marginTop: 4,
        padding: 4,
        maxHeight: 220,
        overflowY: "auto",
      }}
    >
      <div style={{ fontSize: 11, color: "var(--text-faint)", padding: "4px 8px" }}>
        Suggested — grounded in the knowledge base
      </div>
      {suggestions.map((s, i) => (
        <button
          key={i}
          className="ghost"
          style={{
            display: "block",
            width: "100%",
            textAlign: "left",
            padding: "6px 8px",
            fontSize: 13,
          }}
          onClick={() => {
            onPick(s.text);
            setOpen(false);
          }}
        >
          <span className={`badge ${s.kind === "case" ? "case" : "law"}`} style={{ marginRight: 6 }}>
            {s.kind}
          </span>
          {s.text}
          <div style={{ fontSize: 11, color: "var(--text-faint)" }}>{s.detail}</div>
        </button>
      ))}
    </div>
  );
}
