"""Exercise the gateway end to end over HTTP, the same way the UI will.

Usage: python scripts/smoke_api.py [--fir]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"

FIR_TEXT = """
FIRST INFORMATION REPORT
Police Station: Sector 20, Gurugram        FIR No: 0412/2024
Date: 14/03/2024

Complainant: M/s Ravi Traders, through its proprietor Shri Ravi Kumar, Gurugram.
Accused: M/s Sunrise Distributors Pvt. Ltd., through its director Shri Anil Mehta.

Brief facts: The complainant states that under a written supply agreement dated 02/01/2024,
goods (industrial fasteners) worth Rs. 18,40,000 were delivered to the accused on 20/02/2024
against invoice No. RT/2024/118. The accused issued a cheque bearing No. 556213 dated
05/03/2024 for Rs. 18,40,000 drawn on ICICI Bank, which was returned unpaid with the remark
"funds insufficient" on 11/03/2024. Despite telephonic demands and a written notice, the
accused has neither paid the amount nor returned the goods. The complainant alleges that the
accused induced delivery with a dishonest intention of not paying, thereby cheating the
complainant and committing criminal breach of trust.
""".strip()


def post(path: str, payload: dict, timeout: float = 1800) -> dict:
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get(path: str, timeout: float = 120) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as response:
        return json.loads(response.read())


def main(use_fir: bool) -> int:
    health = get("/api/system/health")
    print(f"health: {health['status']}  vectors={health['retrievers']['dense']['vectors']}")

    if use_fir:
        print("\n=== pasting an FIR ===")
        pasted = post("/api/documents/paste", {"text": FIR_TEXT, "filename": "FIR-0412-2024.txt"})
        print(json.dumps(pasted, indent=2))
        question = "What offences and civil remedies arise from this FIR, and find similar cases."
    else:
        question = (
            "My client supplied goods under a written agreement and the buyer's cheque bounced. "
            "What provisions apply and are there similar Indian cases?"
        )

    print(f"\n=== research query ===\n{question}\n")
    started = time.perf_counter()
    result = post("/api/research/query", {"question": question})
    elapsed = time.perf_counter() - started

    print(f"trace_id : {result['trace_id']}")
    print(f"wall     : {elapsed:.1f}s (server {result['total_ms'] / 1000:.1f}s)")

    routing = result.get("routing") or {}
    print("\nROUTER DECISION")
    print(f"  task          : {routing.get('task')}")
    print(f"  input_kind    : {routing.get('input_kind')}")
    print(f"  document_type : {routing.get('document_type')}")
    print(f"  agents        : {routing.get('agents_requested')}")
    print(f"  fallback used : {routing.get('used_fallback')}")
    print(f"  ORIGINAL      : {(routing.get('original_input') or '')[:110]}")
    print(f"  SEARCH QUERY  : {routing.get('search_query')}")
    print(f"  concepts      : {routing.get('legal_concepts')}")

    print("\nAGENTS")
    for key, agent in result["agents"].items():
        ctx = agent["context"]
        print(
            f"  {key:<24} {agent['model']:<22} {agent['latency_ms'] / 1000:6.1f}s "
            f"in={agent['input_tokens']} out={agent['output_tokens']} "
            f"ctx={ctx['included_count']}in/{ctx['excluded_count']}out parse={agent['parse_ok']}"
        )

    print("\nLOOP")
    for it in result["loop_iterations"]:
        print(f"  [{it['iteration']}] {it['agent']:<26} {it['action']}")

    trace = get(f"/api/traces/{result['trace_id']}")
    print(f"\nTRACE: {len(trace['stages'])} stages, {len(trace['model_runs'])} model runs")
    for stage in trace["stages"]:
        print(
            f"  {stage['ordinal']:>2} {stage['stage_key']:<26} {stage['status']:<8} "
            f"{(stage['duration_ms'] or 0):>8.0f} ms  [{stage['service']}]"
        )

    if result.get("legal"):
        print("\nLEGAL:", json.dumps(result["legal"])[:400])
    if result.get("cases"):
        print("\nCASES:", json.dumps(result["cases"])[:400])
    if result.get("draft"):
        print("\nDRAFT:", json.dumps(result["draft"])[:400])
    for warning in result.get("warnings") or []:
        print(f"\n! {warning}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fir", action="store_true", help="paste an FIR first")
    raise SystemExit(main(parser.parse_args().fir))
