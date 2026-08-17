"""Run the full multi-agent pipeline end to end and show what each agent did.

This is the demonstration the professor will see in the UI: retrieval -> document
understanding -> (legal research || similar cases) -> verification loop -> drafting.

Usage:
  python scripts/smoke_orchestrator.py
  python scripts/smoke_orchestrator.py --no-draft      # faster: skip the drafting agent
  python scripts/smoke_orchestrator.py --question "..."
"""

from __future__ import annotations

import argparse
import asyncio
import json

from legalaid.agents.orchestrator import Orchestrator
from legalaid.db.base import create_all
from legalaid.domain.enums import DraftType

DEFAULT_QUESTION = (
    "My client supplied goods under a written agreement and the buyer refused payment "
    "after delivery. What provisions apply to claiming compensation for the loss, and are "
    "there similar Indian cases?"
)


async def main(args: argparse.Namespace) -> int:
    create_all()
    orchestrator = Orchestrator()

    print(f"QUESTION\n  {args.question}\n")
    result = await orchestrator.run(
        question=args.question,
        matter_title="Contract Dispute - Breach and Compensation",
        draft_type=DraftType.CASE_SUMMARY,
        instructions="Summarise the legal position for the file.",
        enable_drafting=not args.no_draft,
    )

    print(f"trace_id : {result.trace_id}")
    print(f"total    : {result.total_ms / 1000:.1f} s\n")

    print("EXECUTION PLAN")
    for step in result.plan:
        marker = "||" if step.get("parallel") else "->"
        print(f"  {marker} {step['step']}  ({step.get('reason') or step.get('mode')})")

    print("\nRETRIEVAL")
    for trace in result.retrieval.searches:
        if trace.returned:
            print(
                f"  {trace.collection.value:<16} {trace.candidate_count:>4} candidates -> "
                f"{trace.returned} used  ({trace.search_ms:.0f} ms)"
            )

    print("\nAGENTS")
    for name, agent_result in result.agent_results.items():
        ctx = agent_result.context
        print(f"\n  {name}")
        print(f"    model    : {agent_result.model}  prompt={agent_result.prompt_label}")
        print(
            f"    tokens   : in={agent_result.input_tokens} out={agent_result.output_tokens}  "
            f"latency={agent_result.latency_ms / 1000:.1f}s  attempts={agent_result.attempts}"
        )
        print(
            f"    context  : {len(ctx.included)} included / {len(ctx.excluded)} EXCLUDED "
            f"({ctx.included_tokens} tokens)"
        )
        kinds: dict[str, int] = {}
        for item in ctx.included:
            kinds[item.kind] = kinds.get(item.kind, 0) + 1
        print(f"    included : {kinds}")
        if ctx.excluded:
            reasons: dict[str, int] = {}
            for item in ctx.excluded:
                reasons[item.reason] = reasons.get(item.reason, 0) + 1
            top = sorted(reasons.items(), key=lambda kv: -kv[1])[:2]
            print(f"    excluded : {top}")
        print(f"    parse_ok : {agent_result.parse_ok}  {agent_result.parse_error or ''}")

    print("\nREFINEMENT LOOP")
    if not result.loop_iterations:
        print("  (no iterations recorded)")
    for it in result.loop_iterations:
        print(
            f"  [{it.iteration}] {it.agent:<28} {it.action:<24} "
            f"problems={it.audit.get('problem_count')}"
        )
        if it.guidance:
            print(f"        guidance: {it.guidance[:150]}")

    print("\n--- LEGAL RESEARCH OUTPUT ---")
    print(json.dumps(result.legal, indent=2)[:1800])

    print("\n--- SIMILAR CASES OUTPUT ---")
    print(json.dumps(result.cases, indent=2)[:1800])

    if result.draft:
        print("\n--- DRAFT ---")
        print(json.dumps(result.draft, indent=2)[:1800])

    if result.warnings:
        print("\nWARNINGS")
        for warning in result.warnings:
            print(f"  ! {warning}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--no-draft", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
