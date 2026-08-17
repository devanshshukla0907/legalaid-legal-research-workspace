"""Verify the pipeline now produces a readable answer, not just structured findings."""

from __future__ import annotations

import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
QUESTION = (
    "My client supplied goods under a written agreement and the buyer refused payment after "
    "delivery. What provisions apply to claiming compensation, and are there similar cases?"
)


def main() -> int:
    body = json.dumps({"question": QUESTION}).encode()
    request = urllib.request.Request(
        f"{BASE}/api/research/query",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=2400) as response:
        result = json.loads(response.read())
    elapsed = time.perf_counter() - started

    print(f"trace   : {result['trace_id']}")
    print(f"wall    : {elapsed:.1f}s")

    answer = result.get("answer") or {}
    if not answer:
        print("\n!! NO ANSWER RETURNED")
        return 1

    print(f"\nconfidence: {answer.get('confidence')}")
    if answer.get("used_fallback"):
        print("!! deterministic fallback was used (synthesis did not complete)")

    print("\n================ ANSWER ================")
    print(answer.get("answer_markdown", "(empty)"))

    for label, key in (
        ("KEY POINTS", "key_points"),
        ("INFERENCE", "inference"),
        ("UNCERTAINTIES", "uncertainties"),
        ("NEXT STEPS", "next_steps"),
        ("AUTHORITIES", "authorities_cited"),
    ):
        items = answer.get(key) or []
        if items:
            print(f"\n{label}")
            for item in items:
                print(f"  - {item}")

    print("\nAGENTS")
    for name, agent in result["agents"].items():
        print(
            f"  {name:<24} {agent['latency_ms'] / 1000:6.1f}s  parse={agent['parse_ok']}  "
            f"out={agent['output_tokens']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
