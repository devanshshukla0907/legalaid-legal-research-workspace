"""Run the guardrail probe suite through the API and print the scorecard (Exercise 13)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
KEY = (Path(__file__).resolve().parents[1] / "data" / ".dev-api-key").read_text().strip()


def probe(check_corpus: bool) -> dict:
    request = urllib.request.Request(
        f"{BASE}/api/guardrails/probe?check_corpus={str(check_corpus).lower()}",
        data=b"{}",
        method="POST",
    )
    request.add_header("Content-Type", "application/json")
    request.add_header("X-API-Key", KEY)
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def report(label: str, payload: dict) -> None:
    summary = payload["summary"]
    print(f"\n=== {label} ===")
    print(f"  adversarial      : {summary['adversarial']}")
    print(f"  caught           : {summary['caught']}  ({summary['catch_rate'] * 100:.1f}%)")
    print(f"  by intended layer: {summary['caught_by_expected_layer']}")
    print(f"  legitimate       : {summary['legitimate']}")
    print(
        f"  false positives  : {summary['false_positives']}  "
        f"({summary['false_positive_rate'] * 100:.1f}%)"
    )
    print(f"  overall accuracy : {summary['accuracy'] * 100:.1f}%")
    print(f"  mean screen cost : {summary['mean_latency_ms']:.0f} ms")
    print(f"  refusals by layer: {summary['blocked_by_layer']}")

    wrong = [c for c in payload["cases"] if not c["correct"]]
    if wrong:
        print("  incorrect:")
        for case in wrong:
            expected = case["expected_blocked_by"] or "allow"
            actual = case["blocked_by"] or "allowed"
            print(f"    {case['id']:<4} expected={expected:<16} actual={actual:<16} "
                  f"{case['text'][:52]}")


def main() -> None:
    full = probe(True)
    report("full stack (with embedding layer)", full)
    deterministic = probe(False)
    report("deterministic layers only", deterministic)

    gain = full["summary"]["catch_rate"] - deterministic["summary"]["catch_rate"]
    print(
        f"\ncorpus-distance layer contributes {gain * 100:+.1f} percentage points of catch rate"
    )

    out = Path(__file__).resolve().parents[1] / "eval" / "results" / "guardrail_suite.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"full": full, "deterministic_only": deterministic}, indent=2),
        encoding="utf-8",
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
