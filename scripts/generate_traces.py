"""Run full-pipeline requests so E12 has real recorded stage timings to aggregate.

The evaluation harness runs a single agent, which does not populate `trace_stages`. The latency
breakdown needs genuine end-to-end runs through the orchestrator, so this drives the real
endpoint and lets the tracing layer record what it normally records.

Requires the gateway to be running.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
KEY = (Path(__file__).resolve().parents[1] / "data" / ".dev-api-key").read_text().strip()

# A spread across the task categories, so the stage mix is representative rather than all
# statutory lookups (which skip document understanding and often drafting).
QUESTIONS = [
    "What does Section 73 of the Indian Contract Act provide for?",
    "My client supplied goods under a written agreement and the buyer refused payment after "
    "delivery. Which provisions support a claim for compensation?",
    "Our contract fixes a sum payable on default. Is that clause enforceable and how much can "
    "actually be recovered?",
    "Our entire agreement and all correspondence exist only as email and PDF. Is the contract "
    "valid, and how do we prove these records in court?",
    "In which court should we institute a suit for breach of a supply contract?",
    "Draft a reply to a legal notice alleging breach of a supply agreement.",
    "Are there Supreme Court judgments on arbitration agreements in unstamped contracts?",
    "The supplier delivered three months late. Was time of the essence, and what follows?",
]


def ask(question: str, *, model: str | None = None, timeout: int = 1500) -> dict:
    body = {"question": question, "enable_verification": True}
    if model:
        body["model"] = model
    request = urllib.request.Request(
        f"{BASE}/api/research/query", data=json.dumps(body).encode(), method="POST"
    )
    request.add_header("Content-Type", "application/json")
    request.add_header("X-API-Key", KEY)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"
    print(f"model: {model}   requests: {len(QUESTIONS)}\n")
    ok = 0
    for index, question in enumerate(QUESTIONS, start=1):
        started = time.perf_counter()
        try:
            result = ask(question, model=model)
        except urllib.error.HTTPError as exc:
            print(f"  [{index}/{len(QUESTIONS)}] HTTP {exc.code}: {exc.read()[:160]!r}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  [{index}/{len(QUESTIONS)}] {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - started
        answer = (result.get("answer") or {})
        guard = result.get("guardrail") or {}
        out_guard = result.get("output_guardrail") or {}
        print(
            f"  [{index}/{len(QUESTIONS)}] {elapsed:>6.1f}s  trace={result.get('trace_id')}  "
            f"agents={len(result.get('agent_results') or {}) or 'n/a'}  "
            f"guard={'allow' if guard.get('allowed', True) else guard.get('blocked_by')}  "
            f"grounded={out_guard.get('groundedness')}  "
            f"conf={answer.get('confidence')}"
        )
        ok += 1
    print(f"\n{ok}/{len(QUESTIONS)} completed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
