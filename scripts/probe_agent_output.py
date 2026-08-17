"""Inspect exactly what Code Llama emitted for a failing agent, from the recorded trace."""

from __future__ import annotations

from sqlalchemy import select

from legalaid.db.base import session_scope
from legalaid.db.models import AgentRun, ModelRun


def main() -> None:
    with session_scope() as session:
        runs = session.scalars(
            select(AgentRun).order_by(AgentRun.started_at.desc()).limit(6)
        ).all()
        for run in runs:
            manifest = run.context_manifest or {}
            print("=" * 96)
            print(
                f"{run.agent_key}  status={run.status}  attempts={manifest.get('attempts')} "
                f"parse_ok={manifest.get('parse_ok')}"
            )
            print(f"  error: {run.error}")
            print(f"  in={manifest.get('input_tokens')} out={manifest.get('output_tokens')}")
            raw = manifest.get("raw_output") or ""
            print(f"  --- RAW MODEL OUTPUT ({len(raw)} chars) ---")
            print(raw[:2200])
            print()

        print("\n\n### recent model runs")
        for mr in session.scalars(
            select(ModelRun).order_by(ModelRun.created_at.desc()).limit(6)
        ).all():
            print(
                f"  {mr.agent_key:<20} {mr.status:<8} in={mr.input_tokens} out={mr.output_tokens} "
                f"{mr.total_duration_ms / 1000:.1f}s tok/s={mr.tokens_per_second}"
            )


if __name__ == "__main__":
    main()
