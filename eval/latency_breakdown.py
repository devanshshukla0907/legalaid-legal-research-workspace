"""E12: where does the time actually go?

The pipeline takes minutes per question, and "it is slow because the model is slow" is not an
analysis. Every stage is already recorded in `trace_stages` with a duration, so this aggregates
recorded runs rather than instrumenting anything new - the numbers come from real requests.

Reports per stage: share of total, mean, median and p95. Median matters because one 90-second
outlier distorts a mean, and p95 is what a user actually experiences as "slow".

Reads the database directly. Run some pipeline requests first, or point it at existing traces.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from sqlalchemy import desc, select

from legalaid.db.base import session_scope
from legalaid.db.models import AgentRun, ModelRun, PipelineTrace, TraceStage
from legalaid.domain.enums import STAGE_LABELS

OUT = Path(__file__).parent / "results" / "latency_breakdown.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=int, default=20, help="how many recent traces to use")
    args = parser.parse_args()

    with session_scope() as session:
        traces = session.scalars(
            select(PipelineTrace).order_by(desc(PipelineTrace.started_at)).limit(args.traces)
        ).all()
        if not traces:
            print("No traces recorded. Run some pipeline requests first.")
            return
        trace_ids = [t.id for t in traces]

        stages = session.scalars(
            select(TraceStage).where(TraceStage.trace_id.in_(trace_ids))
        ).all()
        agent_runs = session.scalars(
            select(AgentRun).where(AgentRun.trace_id.in_(trace_ids))
        ).all()
        model_runs = session.scalars(
            select(ModelRun).where(ModelRun.trace_id.in_(trace_ids))
        ).all()
        totals = [t.total_ms for t in traces if t.total_ms]

    by_stage: dict[str, list[float]] = defaultdict(list)
    for stage in stages:
        if stage.duration_ms:
            by_stage[stage.stage_key].append(stage.duration_ms)

    by_agent: dict[str, list[float]] = defaultdict(list)
    for run in agent_runs:
        if run.duration_ms:
            by_agent[run.agent_label or run.agent_key].append(run.duration_ms)

    grand_total = sum(totals) if totals else sum(sum(v) for v in by_stage.values())

    print(f"traces analysed : {len(traces)}")
    print(f"stages recorded : {len(stages)}")
    print(f"agent runs      : {len(agent_runs)}")
    print(f"model calls     : {len(model_runs)}")
    if totals:
        print(f"mean total      : {statistics.fmean(totals) / 1000:.1f} s")
        print(f"median total    : {statistics.median(totals) / 1000:.1f} s")

    def table(title: str, data: dict[str, list[float]], denominator: float) -> list[dict]:
        print(f"\n=== {title} ===")
        print(f"{'stage':<40}{'n':>4}{'share':>8}{'mean':>10}{'median':>10}{'p95':>10}")
        print("-" * 82)
        rows = []
        for key, values in sorted(data.items(), key=lambda kv: -sum(kv[1])):
            ordered = sorted(values)
            p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
            share = sum(values) / denominator if denominator else 0.0
            label = STAGE_LABELS.get(key, key) if title.startswith("stage") else key
            rows.append(
                {
                    "key": key,
                    "label": str(label),
                    "count": len(values),
                    "share_of_total": round(share, 4),
                    "total_ms": round(sum(values), 1),
                    "mean_ms": round(statistics.fmean(values), 1),
                    "median_ms": round(statistics.median(values), 1),
                    "p95_ms": round(p95, 1),
                }
            )
            print(
                f"{str(label)[:39]:<40}{len(values):>4}{share * 100:>7.1f}%"
                f"{statistics.fmean(values) / 1000:>9.2f}s"
                f"{statistics.median(values) / 1000:>9.2f}s{p95 / 1000:>9.2f}s"
            )
        return rows

    stage_rows = table("stage breakdown", by_stage, grand_total)
    agent_rows = table("agent breakdown", by_agent, grand_total)

    # Generation versus everything else. This is the number that decides whether optimisation
    # effort belongs in the pipeline or in the model.
    generation_ms = sum(r.total_duration_ms or 0 for r in model_runs)
    print("\n=== generation vs orchestration ===")
    if grand_total:
        print(f"  total pipeline      : {grand_total / 1000:.1f} s")
        print(f"  in model generation : {generation_ms / 1000:.1f} s "
              f"({generation_ms / grand_total * 100:.1f}%)")
        print(f"  everything else     : {(grand_total - generation_ms) / 1000:.1f} s "
              f"({(grand_total - generation_ms) / grand_total * 100:.1f}%)")
        print(
            "\n  Generation dominating means orchestration overhead is not the problem to "
            "solve;\n  a faster model or fewer agent calls is."
        )

    tokens = [r.output_tokens for r in model_runs if r.output_tokens]
    rates = [r.tokens_per_second for r in model_runs if r.tokens_per_second]
    if rates:
        print(f"\n  mean throughput     : {statistics.fmean(rates):.1f} tok/s")
        print(f"  total output tokens : {sum(tokens)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "traces_analysed": len(traces),
                "mean_total_ms": round(statistics.fmean(totals), 1) if totals else None,
                "median_total_ms": round(statistics.median(totals), 1) if totals else None,
                "grand_total_ms": round(grand_total, 1),
                "generation_ms": round(generation_ms, 1),
                "generation_share": round(generation_ms / grand_total, 4) if grand_total else None,
                "mean_tokens_per_second": round(statistics.fmean(rates), 2) if rates else None,
                "stages": stage_rows,
                "agents": agent_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
