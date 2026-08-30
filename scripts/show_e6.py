"""Print the E6 RAG on/off comparison table."""

from __future__ import annotations

import json
from pathlib import Path

data = json.loads(
    (Path(__file__).resolve().parents[1] / "eval" / "results" / "rag_ablation.json").read_text()
)
no_rag, with_rag = data["no_rag"], data["with_rag"]

print(f"model: {data['model']}\n")
print(f"{'metric':<34}{'no RAG':>12}{'with RAG':>12}")
print("-" * 58)
for key in (
    "sections_named",
    "corpus_verified",
    "fabricated",
    "fabrication_rate",
    "verified_rate",
    "tasks_with_a_fabrication",
    "accuracy_answerable",
    "declined_on_unanswerable",
    "fabricated_on_unanswerable",
    "mean_latency_ms",
    "mean_output_tokens",
):
    left, right = no_rag[key], with_rag[key]
    if isinstance(left, float):
        print(f"{key:<34}{left:>12.3f}{right:>12.3f}")
    else:
        print(f"{key:<34}{left:>12}{right:>12}")

print("\n--- fabrications in the no-RAG condition ---")
for record in data["results"]:
    if record["condition"] == "no_rag" and record["fabricated"]:
        print(f"  {record['task_id']}: {record['fabricated']}")

print("\n--- fabrications in the with-RAG condition ---")
for record in data["results"]:
    if record["condition"] == "with_rag" and record["fabricated"]:
        print(f"  {record['task_id']}: {record['fabricated']}")
