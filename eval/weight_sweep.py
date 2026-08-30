"""Find the keyword fusion weight by measurement instead of intuition.

E5 established that the full hybrid stack retrieves *worse* than dense alone: recall 0.568
against 0.614, at twice the latency. The cause is in how Reciprocal Rank Fusion rewards
agreement. A document both retrievers found accumulates both contributions, so a document dense
ranked 20th but keyword ranked 1st can outscore one dense ranked 5th:

    dense rank 5  only          1/(60+5)                 = 0.01538
    dense rank 20 + kw rank 1   1/(60+20) + 0.6/(60+1)   = 0.02234

That is exactly what RRF is designed to do, and it is the right behaviour when both retrievers
are strong. On a 2575-chunk legal corpus BM25 is not: without an explicit section number in the
question it matches on generic legal vocabulary - "contract", "party", "compensation" - and
those spurious agreements promote noise over good semantic hits.

`retrieval_keyword_weight = 0.6` was chosen by reasoning ("semantic is the primary signal, so
weight BM25 lower"). The reasoning was sound and the number was still wrong. This sweep measures
it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from dataset import TASKS, EvalTask
from legalaid.domain.contracts import RetrievalRequest
from legalaid.domain.enums import RetrievalMode
from legalaid.services.retrieval.service import search
from runner import _section_key

OUT = Path(__file__).parent / "results" / "weight_sweep.json"

WEIGHTS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]


async def score(task: EvalTask, keyword_weight: float, *, graph: bool) -> tuple[float, float]:
    expected = [_section_key(act, num) for act, num in task.expected_sections]
    response = await search(
        RetrievalRequest(
            question=task.question,
            mode=RetrievalMode.COMBINED_LEGAL_RESEARCH,
            top_k=8,
            enable_dense=True,
            enable_keyword=keyword_weight > 0,
            enable_graph=graph,
            keyword_weight=keyword_weight,
        )
    )
    retrieved = [
        _section_key(c.source_title or "", str(c.section))
        for c in response.chunks
        if c.section
    ]
    found = [k for k in expected if k in retrieved]
    recall = len(found) / len(expected) if expected else 0.0
    mrr = 0.0
    for rank, key in enumerate(retrieved, start=1):
        if key in expected:
            mrr = 1.0 / rank
            break
    return recall, mrr


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="store_true", help="also enable the graph retriever")
    args = parser.parse_args()

    tasks = [t for t in TASKS if t.expected_sections]
    print(f"{len(WEIGHTS)} weights x {len(tasks)} tasks (graph={'on' if args.graph else 'off'})\n")
    print(f"{'kw weight':>10}{'recall':>9}{'mrr':>8}{'zero':>7}{'perfect':>9}")
    print("-" * 45)

    rows = []
    for weight in WEIGHTS:
        recalls: list[float] = []
        mrrs: list[float] = []
        for task in tasks:
            recall, mrr = await score(task, weight, graph=args.graph)
            recalls.append(recall)
            mrrs.append(mrr)
        row = {
            "keyword_weight": weight,
            "mean_recall": round(statistics.fmean(recalls), 4),
            "mean_mrr": round(statistics.fmean(mrrs), 4),
            "zero_recall": sum(1 for r in recalls if r == 0.0),
            "perfect_recall": sum(1 for r in recalls if r == 1.0),
        }
        rows.append(row)
        print(
            f"{weight:>10.2f}{row['mean_recall']:>9.3f}{row['mean_mrr']:>8.3f}"
            f"{row['zero_recall']:>7}{row['perfect_recall']:>9}"
        )

    best_recall = max(rows, key=lambda r: (r["mean_recall"], r["mean_mrr"]))
    dense_only = next(r for r in rows if r["keyword_weight"] == 0.0)

    print(f"\nbest        : weight {best_recall['keyword_weight']} "
          f"recall {best_recall['mean_recall']:.3f} mrr {best_recall['mean_mrr']:.3f}")
    print(f"dense only  : recall {dense_only['mean_recall']:.3f} "
          f"mrr {dense_only['mean_mrr']:.3f}")
    gain = best_recall["mean_recall"] - dense_only["mean_recall"]
    print(f"keyword worth: {gain:+.4f} recall at its best weight")
    if gain <= 0:
        print("  -> BM25 does not help at any weight on this corpus. Honest answer: turn it off,")
        print("     or restrict it to queries that name a section explicitly.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "graph_enabled": args.graph,
                "rows": rows,
                "best": best_recall,
                "dense_only": dense_only,
                "keyword_marginal_recall": round(gain, 4),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
