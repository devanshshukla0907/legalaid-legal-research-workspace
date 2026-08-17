"""Test one agent in isolation with constrained decoding, to verify schema compliance fast."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from legalaid.agents import legal_research as lr
from legalaid.agents import similar_cases as sc
from legalaid.agents.legal_research import LegalResearchAgent
from legalaid.agents.similar_cases import SimilarCaseAgent
from legalaid.agents.verification import audit
from legalaid.db.base import create_all
from legalaid.domain.contracts import RetrievalRequest
from legalaid.domain.enums import RetrievalMode
from legalaid.services.retrieval.service import search

QUESTION = (
    "My client supplied goods under a written agreement and the buyer refused payment. "
    "What provisions apply to claiming compensation for the loss?"
)


async def main(which: str) -> int:
    create_all()
    mode = (
        RetrievalMode.CASE_SEARCH if which == "cases" else RetrievalMode.LAW_SEARCH
    )
    retrieval = await search(RetrievalRequest(question=QUESTION, mode=mode, top_k=6))
    print(f"retrieved {len(retrieval.chunks)} chunks")

    if which == "cases":
        agent = SimilarCaseAgent()
        context = sc.prepare(agent, question=QUESTION, chunks=retrieval.chunks)
    else:
        agent = LegalResearchAgent()
        context = lr.prepare(agent, question=QUESTION, chunks=retrieval.chunks)

    print(f"context: {len(context.included)} included, {len(context.excluded)} excluded")
    started = time.perf_counter()
    result = await agent.run(trace_id=None, context=context, max_attempts=1)
    elapsed = time.perf_counter() - started

    print(f"\nlatency   : {elapsed:.1f}s")
    print(f"tokens    : in={result.input_tokens} out={result.output_tokens}")
    print(f"parse_ok  : {result.parse_ok}  {result.parse_error or ''}")
    print(f"attempts  : {result.attempts}")

    problems = agent.validate(result.output, context) if result.parse_ok else ["not parsed"]
    report = audit(
        agent_name=which, output=result.output, context=context, structural_problems=problems
    )
    print(f"audit     : passed={report.passed} problems={report.problem_count}")
    if not report.passed:
        print(f"  guidance: {report.guidance()[:300]}")

    print("\n--- OUTPUT ---")
    print(json.dumps(result.output, indent=2)[:2500])
    return 0 if result.parse_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="law", choices=["law", "cases"])
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.which)))
