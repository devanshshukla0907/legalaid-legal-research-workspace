"""Re-verify every expected section in the evaluation dataset against the live corpus.

The corpus grew from 263 to 1783 provisions, so this must be re-run: an expected section that
no longer resolves would score every model as wrong for a reason that has nothing to do with the
model. Equally, sections that were absent before may now be present, which can turn a
deliberately-unanswerable task into an answerable one and invalidate the hallucination metric.
"""

from __future__ import annotations

import sys

from sqlalchemy import select

from dataset import TASKS
from legalaid.db.base import session_scope
from legalaid.db.models import LegalProvision


def main() -> int:
    with session_scope() as session:
        rows = session.scalars(select(LegalProvision)).all()

    # Act names are matched on a normalised prefix, the same way the harness does it.
    available: dict[str, set[str]] = {}
    for row in rows:
        token = "".join(ch for ch in (row.act_name or "").lower() if ch.isalnum())[:14]
        available.setdefault(token, set()).add((row.section_number or "").upper())

    print(f"corpus: {len(rows)} provisions across {len(available)} acts\n")

    missing: list[str] = []
    print(f"{'task':<6}{'expected':<46}{'status'}")
    print("-" * 74)
    for task in TASKS:
        if not task.expected_sections:
            continue
        for act, number in task.expected_sections:
            token = "".join(ch for ch in act.lower() if ch.isalnum())[:14]
            present = number.upper() in available.get(token, set())
            label = f"{act} s.{number}"
            print(f"{task.id:<6}{label[:45]:<46}{'OK' if present else 'MISSING'}")
            if not present:
                missing.append(f"{task.id}: {label}")

    print("\n=== unanswerable tasks: are they still unanswerable? ===")
    # These name subjects the corpus should not contain. If one is now present the task must be
    # reclassified, or the hallucination metric silently measures the wrong thing.
    probes = {
        "N01": ("indianpenalco", "498A", "Indian Penal Code s.498A"),
        "N02": ("centralgoodsa", "22", "GST registration threshold"),
        "N03": ("motorvehicles", "161", "Motor Vehicles Act hit-and-run"),
    }
    for task_id, (token, number, label) in probes.items():
        present = number.upper() in available.get(token, set())
        status = "NOW PRESENT - RECLASSIFY" if present else "still absent, task valid"
        print(f"  {task_id}  {label:<44} {status}")

    print(f"\nacts in corpus: {', '.join(sorted(available))}")

    if missing:
        print(f"\n{len(missing)} expected section(s) missing:")
        for entry in missing:
            print(f"  - {entry}")
        return 1
    print("\nAll expected sections resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
