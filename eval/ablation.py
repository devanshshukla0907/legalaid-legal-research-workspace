"""E5: retriever ablation. What does each retriever actually contribute?

The application claims hybrid retrieval - dense embeddings, BM25 keyword search and a symbolic
graph traversal, fused by reciprocal rank. That claim is architecture, not evidence. This runs
the same 27 questions under all seven combinations of the three retrievers and measures what
each one is worth.

Retrieval-only, so no generation cost: 7 configurations x 27 questions is 189 retrievals, and
the embedding is the only expensive part.

Metrics, all model-independent:

recall
    Of the sections a task is known to need, the fraction retrieved anywhere in top-k.
MRR
    1 / rank of the first expected section. Rewards ranking the right provision first rather
    than merely inside the window, which is what matters when an agent only sees the top few.
precision@k
    Fraction of retrieved chunks that are an expected section. Low precision is not necessarily
    bad here - retrieving a related provision is often useful - but a collapse in precision with
    no recall gain is a real cost.
unique_acts
    How many distinct acts the results span. A retriever that returns eight chunks of one
    section has less diverse evidence than one returning four sections across two acts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dataset import TASKS, EvalTask
from legalaid.domain.contracts import RetrievalRequest
from legalaid.domain.enums import RetrievalMode
from legalaid.services.retrieval.service import search
from runner import _section_key

OUT = Path(__file__).parent / "results" / "ablation.json"

# All seven non-empty combinations. Naming keeps the single-retriever runs first so the table
# reads from "one signal alone" to "everything".
CONFIGS: list[tuple[str, dict[str, bool]]] = [
    ("dense", {"dense": True, "keyword": False, "graph": False}),
    ("keyword", {"dense": False, "keyword": True, "graph": False}),
    ("graph", {"dense": False, "keyword": False, "graph": True}),
    ("dense+keyword", {"dense": True, "keyword": True, "graph": False}),
    ("dense+graph", {"dense": True, "keyword": False, "graph": True}),
    ("keyword+graph", {"dense": False, "keyword": True, "graph": True}),
    ("all three", {"dense": True, "keyword": True, "graph": True}),
]


@dataclass
class TaskResult:
    task_id: str
    recall: float
    mrr: float
    precision: float
    returned: int
    unique_acts: int
    latency_ms: float
    expected: list[str] = field(default_factory=list)
    found: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    top_sections: list[str] = field(default_factory=list)


@dataclass
class ConfigResult:
    name: str
    dense: bool
    keyword: bool
    graph: bool
    tasks: list[TaskResult] = field(default_factory=list)

    def _mean(self, attr: str) -> float:
        values = [getattr(t, attr) for t in self.tasks]
        return round(statistics.fmean(values), 4) if values else 0.0

    @property
    def summary(self) -> dict[str, Any]:
        recalls = [t.recall for t in self.tasks]
        return {
            "name": self.name,
            "dense": self.dense,
            "keyword": self.keyword,
            "graph": self.graph,
            "mean_recall": self._mean("recall"),
            "mean_mrr": self._mean("mrr"),
            "mean_precision": self._mean("precision"),
            "mean_returned": self._mean("returned"),
            "mean_unique_acts": self._mean("unique_acts"),
            "mean_latency_ms": self._mean("latency_ms"),
            "perfect_recall": sum(1 for r in recalls if r == 1.0),
            "zero_recall": sum(1 for r in recalls if r == 0.0),
            "tasks_scored": len(self.tasks),
        }


async def run_one(task: EvalTask, config: dict[str, bool]) -> TaskResult:
    expected = [_section_key(act, num) for act, num in task.expected_sections]
    started = time.perf_counter()
    response = await search(
        RetrievalRequest(
            question=task.question,
            mode=RetrievalMode.COMBINED_LEGAL_RESEARCH,
            top_k=8,
            enable_dense=config["dense"],
            enable_keyword=config["keyword"],
            enable_graph=config["graph"],
        )
    )
    elapsed = (time.perf_counter() - started) * 1000

    retrieved: list[str] = []
    acts: set[str] = set()
    labels: list[str] = []
    for chunk in response.chunks:
        act = chunk.source_title or ""
        section = chunk.section or ""
        if act:
            acts.add(act)
        if section:
            retrieved.append(_section_key(act, str(section)))
            labels.append(f"{section} ({act[:26]})")

    found = [k for k in expected if k in retrieved]
    missed = [k for k in expected if k not in retrieved]
    mrr = 0.0
    for rank, key in enumerate(retrieved, start=1):
        if key in expected:
            mrr = 1.0 / rank
            break

    return TaskResult(
        task_id=task.id,
        recall=round(len(found) / len(expected), 4) if expected else 0.0,
        mrr=round(mrr, 4),
        precision=round(
            sum(1 for k in retrieved if k in expected) / max(1, len(retrieved)), 4
        )
        if expected
        else 0.0,
        returned=len(response.chunks),
        unique_acts=len(acts),
        latency_ms=round(elapsed, 1),
        expected=expected,
        found=found,
        missed=missed,
        top_sections=labels[:5],
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Only tasks with known expected sections can be scored for recall.
    scored = [t for t in TASKS if t.expected_sections]
    if args.limit:
        scored = scored[: args.limit]

    print(f"{len(CONFIGS)} configurations x {len(scored)} scorable tasks "
          f"= {len(CONFIGS) * len(scored)} retrievals\n")

    results: list[ConfigResult] = []
    for name, config in CONFIGS:
        result = ConfigResult(
            name=name, dense=config["dense"], keyword=config["keyword"], graph=config["graph"]
        )
        for task in scored:
            result.tasks.append(await run_one(task, config))
        results.append(result)
        summary = result.summary
        print(
            f"  {name:<16} recall={summary['mean_recall']:.3f} "
            f"mrr={summary['mean_mrr']:.3f} prec={summary['mean_precision']:.3f} "
            f"acts={summary['mean_unique_acts']:.1f} "
            f"{summary['mean_latency_ms']:>7.0f}ms  "
            f"zero_recall={summary['zero_recall']}"
        )

    # ---------------------------------------------------------------- analysis
    baseline = next(r for r in results if r.name == "dense")
    full = next(r for r in results if r.name == "all three")

    print("\n=== marginal contribution over dense alone ===")
    contributions = []
    for result in results:
        if result.name == "dense":
            continue
        delta_recall = result.summary["mean_recall"] - baseline.summary["mean_recall"]
        delta_mrr = result.summary["mean_mrr"] - baseline.summary["mean_mrr"]
        contributions.append(
            {
                "config": result.name,
                "delta_recall": round(delta_recall, 4),
                "delta_mrr": round(delta_mrr, 4),
            }
        )
        print(f"  {result.name:<16} recall {delta_recall:+.3f}   mrr {delta_mrr:+.3f}")

    # Where does each retriever alone succeed that the others fail? This is the interesting
    # part: a retriever earns its place by covering questions the others miss, not by having a
    # good average.
    print("\n=== tasks rescued by a single retriever ===")
    per_config_zero = {
        r.name: {t.task_id for t in r.tasks if t.recall == 0.0} for r in results
    }
    rescues: dict[str, list[str]] = {}
    for name in ("keyword", "graph", "dense"):
        others = [n for n in ("keyword", "graph", "dense") if n != name]
        # Tasks this retriever gets and every other single retriever misses.
        got = {t.task_id for t in next(r for r in results if r.name == name).tasks if t.recall > 0}
        missed_by_others = set.intersection(*[per_config_zero[o] for o in others])
        unique = sorted(got & missed_by_others)
        rescues[name] = unique
        print(f"  {name:<10} uniquely retrieves: {unique or 'nothing the others miss'}")

    print("\n=== full stack vs best single retriever ===")
    best_single = max(
        (r for r in results if r.name in {"dense", "keyword", "graph"}),
        key=lambda r: r.summary["mean_recall"],
    )
    print(f"  best single : {best_single.name} recall={best_single.summary['mean_recall']:.3f}")
    print(f"  all three   : recall={full.summary['mean_recall']:.3f} "
          f"({full.summary['mean_recall'] - best_single.summary['mean_recall']:+.3f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "configs": [
                    {"summary": r.summary, "tasks": [asdict(t) for t in r.tasks]}
                    for r in results
                ],
                "contributions": contributions,
                "unique_rescues": rescues,
                "best_single": best_single.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
