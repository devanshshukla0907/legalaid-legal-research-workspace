"""Does query rewriting actually fix the missed retrievals? A controlled A/B.

Exercise 5 claims that the tasks with zero retrieval recall failed because conversational
phrasing dilutes the embedding, and that the intake router's rewrite is what prevents this in
the deployed app. That is a testable claim, not an excuse, so this script tests it.

Method: for each task that retrieval missed, embed (A) the raw question as the harness sent it,
and (B) a rewritten form in dense legal language of the kind the intake router produces. Same
corpus, same retriever, same top_k. Only the query text changes.

Retrieval-only, so it costs no generation time.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from legalaid.domain.contracts import RetrievalRequest
from legalaid.domain.enums import RetrievalMode
from legalaid.services.retrieval.service import search

from dataset import TASKS
from runner import _section_key

OUT = Path(__file__).parent / "results" / "rewrite_experiment.json"

# Rewrites in the style the intake router produces: the legal concepts, stripped of narrative.
# Written without reference to the expected sections, to avoid smuggling the answer in.
REWRITES: dict[str, str] = {
    "D01": (
        "Cheating and criminal breach of trust by dishonestly inducing delivery of goods "
        "without intention to pay; dishonour of cheque; entrustment and misappropriation of "
        "property."
    ),
    "D02": (
        "Measure of compensation for breach of contract; remoteness of damage; liquidated "
        "damages and penalty stipulated in the contract; reasonable compensation."
    ),
    "D03": (
        "Definition of agent and principal; authority of an agent to bind the principal; "
        "contracts entered into through an agent."
    ),
    "R05": (
        "Effect of failure to perform a contract at the fixed time; whether time is of the "
        "essence; voidable contract on delayed performance."
    ),
    "P01": (
        "Court in which a suit is to be instituted; place of institution where the cause of "
        "action arises; jurisdiction over defendants."
    ),
    "X03": (
        "Bailment; duty of care to be taken by a bailee of goods; liability of bailee for loss "
        "or damage to the thing bailed."
    ),
}


async def retrieve_keys(question: str) -> tuple[list[str], list[float], list[str]]:
    response = await search(
        RetrievalRequest(
            question=question, mode=RetrievalMode.COMBINED_LEGAL_RESEARCH, top_k=8
        )
    )
    keys: list[str] = []
    labels: list[str] = []
    for chunk in response.chunks:
        section = chunk.section or ""
        if section:
            keys.append(_section_key(chunk.source_title or "", str(section)))
            labels.append(f"{section} ({chunk.source_title or ''})"[:52])
    return keys, [round(c.similarity, 4) for c in response.chunks], labels


def score(keys: list[str], expected: list[str]) -> tuple[float, float]:
    found = [k for k in expected if k in keys]
    recall = len(found) / len(expected) if expected else 0.0
    mrr = 0.0
    for rank, key in enumerate(keys, start=1):
        if key in expected:
            mrr = 1.0 / rank
            break
    return round(recall, 4), round(mrr, 4)


async def main() -> None:
    tasks = {t.id: t for t in TASKS}
    results = []

    print(f"{'task':<6}{'raw recall':>12}{'raw mrr':>10}{'rw recall':>12}{'rw mrr':>10}  verdict")
    print("-" * 78)

    for task_id, rewrite in REWRITES.items():
        task = tasks[task_id]
        expected = [_section_key(a, n) for a, n in task.expected_sections]
        if not expected:
            continue

        raw_keys, raw_sims, raw_labels = await retrieve_keys(task.question)
        rw_keys, rw_sims, rw_labels = await retrieve_keys(rewrite)

        raw_recall, raw_mrr = score(raw_keys, expected)
        rw_recall, rw_mrr = score(rw_keys, expected)

        if rw_recall > raw_recall:
            verdict = "REWRITE FIXES IT"
        elif rw_recall < raw_recall:
            verdict = "rewrite hurt"
        elif rw_mrr > raw_mrr:
            verdict = "better ranking"
        else:
            verdict = "no change"

        print(
            f"{task_id:<6}{raw_recall:>12.2f}{raw_mrr:>10.2f}"
            f"{rw_recall:>12.2f}{rw_mrr:>10.2f}  {verdict}"
        )

        results.append(
            {
                "task_id": task_id,
                "question_raw": task.question,
                "question_rewritten": rewrite,
                "expected": expected,
                "raw": {
                    "recall": raw_recall,
                    "mrr": raw_mrr,
                    "top_sections": raw_labels[:5],
                    "top_similarity": raw_sims[0] if raw_sims else None,
                },
                "rewritten": {
                    "recall": rw_recall,
                    "mrr": rw_mrr,
                    "top_sections": rw_labels[:5],
                    "top_similarity": rw_sims[0] if rw_sims else None,
                },
                "verdict": verdict,
            }
        )

    improved = sum(1 for r in results if r["rewritten"]["recall"] > r["raw"]["recall"])
    mean_raw = sum(r["raw"]["recall"] for r in results) / len(results)
    mean_rw = sum(r["rewritten"]["recall"] for r in results) / len(results)

    print()
    print(f"tasks tested        : {len(results)}")
    print(f"recall improved on  : {improved}")
    print(f"mean recall raw     : {mean_raw:.3f}")
    print(f"mean recall rewrite : {mean_rw:.3f}")
    delta = mean_rw - mean_raw
    print(f"absolute gain       : {delta:+.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "results": results,
                "summary": {
                    "n": len(results),
                    "improved": improved,
                    "mean_recall_raw": round(mean_raw, 4),
                    "mean_recall_rewritten": round(mean_rw, 4),
                    "absolute_gain": round(delta, 4),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
