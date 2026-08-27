"""Metric definitions for the model comparison.

Every metric here is computed by set membership or arithmetic on recorded run data - none of it
is a subjective read of the output. That matters because the whole point of the exercise is a
conclusion based on quantitative evidence.

METRIC DEFINITIONS
------------------
Retrieval recall
    Of the sections a task is known to need, the fraction that appeared anywhere in the
    retrieved context.  recall = |expected AND retrieved| / |expected|
    Model-independent: retrieval runs once and is replayed to every model.

Retrieval MRR
    1 / rank of the first expected section in the ranked list, 0 if absent. Rewards putting the
    right provision near the top rather than merely somewhere in the top-k.

Correctness (accuracy)
    An answerable task is correct when the model cites at least one expected section AND the
    citation is grounded (its chunk_id was actually supplied). Citing the right section number
    while inventing its source is not counted as correct - in legal research an unverifiable
    citation is worse than no citation.
    An unanswerable task is correct when the model sets evidence_sufficient=false.

Relevance
    Precision of the model's citations against the expected set, averaged over answerable
    tasks:  |cited AND expected| / |cited|. Distinguishes a focused answer from one that lists
    every provision it saw and happens to include the right one.

Hallucination rate
    Fraction of tasks where the model cited at least one chunk_id that was NOT in its supplied
    context. This is the strictest available definition and it is fully automatic: the grounding
    audit knows exactly which chunk_ids the agent was given.
    Reported separately for answerable and unanswerable tasks, because inventing an authority
    when the corpus has nothing is the more serious failure.

Schema compliance ("test-pass rate" analogue)
    Fraction of tasks whose output parsed as JSON AND satisfied the agent's structural
    validator (every cited section carries a resolvable chunk_id). This application does not
    generate code, so a literal unit-test-pass rate does not apply; schema compliance plus
    citation grounding is the equivalent objective pass/fail gate, and it is what the pipeline
    actually depends on.

Latency
    Wall-clock ms for the agent call: prompt render, HTTP to Ollama, generation, parse, and any
    retry. Retrieval is excluded because it is shared and model-independent.

Token usage
    input_tokens and output_tokens as reported by Ollama. Input is near-identical across models
    by construction (same prompt, same context), so differences reflect tokeniser variation;
    output differences reflect verbosity.

Throughput
    output_tokens / (latency_ms / 1000). The fair speed comparison, since a model that writes
    more is not necessarily slower per token.

Resource use
    Sampled separately (eval/resources.py) because Ollama loads one model at a time: peak RSS of
    the ollama process and GPU/VRAM while each model is resident.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset import TASKS

RESULTS = Path(__file__).parent / "results"


@dataclass
class ModelScores:
    model: str
    n: int = 0

    # quality
    correct: int = 0
    correct_answerable: int = 0
    correct_negative: int = 0
    relevance_sum: float = 0.0
    relevance_n: int = 0
    hallucinated: int = 0
    hallucinated_negative: int = 0
    schema_ok: int = 0
    parse_ok: int = 0
    failed: int = 0
    retries: int = 0

    # performance
    latencies: list[float] = field(default_factory=list)
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)

    # Counted from records actually scored, not from the full dataset, so a partially
    # completed run reports coherent percentages instead of mixing denominators.
    seen_answerable: int = 0
    seen_negative: int = 0

    per_task: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ----------------------------------------------------------------- derived
    @property
    def n_answerable(self) -> int:
        return self.seen_answerable

    @property
    def n_negative(self) -> int:
        return self.seen_negative

    @property
    def complete(self) -> bool:
        return self.n == len(TASKS)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def accuracy_answerable(self) -> float:
        return self.correct_answerable / self.n_answerable if self.n_answerable else 0.0

    @property
    def accuracy_negative(self) -> float:
        return self.correct_negative / self.n_negative if self.n_negative else 0.0

    @property
    def relevance(self) -> float:
        return self.relevance_sum / self.relevance_n if self.relevance_n else 0.0

    @property
    def hallucination_rate(self) -> float:
        return self.hallucinated / self.n if self.n else 0.0

    @property
    def schema_compliance(self) -> float:
        return self.schema_ok / self.n if self.n else 0.0

    @property
    def p50_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def mean_latency(self) -> float:
        return statistics.fmean(self.latencies) if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]

    @property
    def mean_output_tokens(self) -> float:
        return statistics.fmean(self.output_tokens) if self.output_tokens else 0.0

    @property
    def mean_input_tokens(self) -> float:
        return statistics.fmean(self.input_tokens) if self.input_tokens else 0.0

    @property
    def throughput(self) -> float:
        """Output tokens per second, over the whole run."""
        total_s = sum(self.latencies) / 1000
        return (sum(self.output_tokens) / total_s) if total_s else 0.0


def _norm(value: str) -> str:
    return value.strip().lower()


def score_model(model_file: Path, retrieval: dict[str, Any]) -> ModelScores:
    records = json.loads(model_file.read_text(encoding="utf-8"))
    tasks = {t.id: t for t in TASKS}
    model_name = records[0]["model"] if records else model_file.stem
    scores = ModelScores(model=model_name)

    for record in records:
        task = tasks.get(record["task_id"])
        if task is None:
            continue
        scores.n += 1
        if task.answerable:
            scores.seen_answerable += 1
        else:
            scores.seen_negative += 1
        supplied = set(retrieval[task.id]["chunk_ids"])
        cited_chunks = set(record.get("cited_chunk_ids") or [])
        cited_sections = {_norm(s) for s in (record.get("cited_sections") or [])}
        expected = {
            _norm(k) for k in _expected_keys_for(task)
        }

        if record.get("error"):
            scores.failed += 1
        if record.get("parse_ok"):
            scores.parse_ok += 1
        scores.retries += max(0, (record.get("attempts") or 1) - 1)

        # ---- hallucination: any citation whose chunk_id was never supplied
        ungrounded = cited_chunks - supplied
        hallucinated = bool(ungrounded)
        if hallucinated:
            scores.hallucinated += 1
            if not task.answerable:
                scores.hallucinated_negative += 1

        # ---- schema compliance: parsed AND every citation grounded
        schema_ok = bool(record.get("parse_ok")) and not ungrounded
        if schema_ok:
            scores.schema_ok += 1

        # ---- correctness
        correct = False
        if task.answerable:
            hit = cited_sections & expected
            correct = bool(hit) and not ungrounded
            if correct:
                scores.correct_answerable += 1
            if cited_sections:
                scores.relevance_sum += len(cited_sections & expected) / len(cited_sections)
                scores.relevance_n += 1
        else:
            # Correct behaviour on an unanswerable task is to decline.
            declined = record.get("evidence_sufficient") is False or not cited_sections
            correct = declined and not ungrounded
            if correct:
                scores.correct_negative += 1
        if correct:
            scores.correct += 1

        if record.get("latency_ms"):
            scores.latencies.append(float(record["latency_ms"]))
        if record.get("input_tokens"):
            scores.input_tokens.append(int(record["input_tokens"]))
        if record.get("output_tokens"):
            scores.output_tokens.append(int(record["output_tokens"]))

        scores.per_task[task.id] = {
            "category": task.category,
            "answerable": task.answerable,
            "correct": correct,
            "hallucinated": hallucinated,
            "ungrounded_chunks": sorted(ungrounded),
            "schema_ok": schema_ok,
            "cited": sorted(cited_sections),
            "expected": sorted(expected),
            "latency_ms": record.get("latency_ms"),
            "output_tokens": record.get("output_tokens"),
            "evidence_sufficient": record.get("evidence_sufficient"),
        }

    return scores


def _expected_keys_for(task) -> list[str]:
    from runner import _section_key

    return [_section_key(act, num) for act, num in task.expected_sections]


def retrieval_summary(retrieval: dict[str, Any]) -> dict[str, float]:
    answerable = [t for t in TASKS if t.answerable and t.expected_sections]
    recalls = [retrieval[t.id]["recall"] for t in answerable if t.id in retrieval]
    mrrs = [retrieval[t.id]["mrr"] for t in answerable if t.id in retrieval]
    latencies = [retrieval[t.id]["retrieval_ms"] for t in TASKS if t.id in retrieval]
    return {
        "n_scored": len(recalls),
        "recall": statistics.fmean(recalls) if recalls else 0.0,
        "mrr": statistics.fmean(mrrs) if mrrs else 0.0,
        "perfect_recall_pct": (sum(1 for r in recalls if r == 1.0) / len(recalls))
        if recalls
        else 0.0,
        "zero_recall_pct": (sum(1 for r in recalls if r == 0.0) / len(recalls))
        if recalls
        else 0.0,
        "mean_ms": statistics.fmean(latencies) if latencies else 0.0,
    }


def load_all() -> tuple[dict[str, Any], list[ModelScores]]:
    retrieval = json.loads((RESULTS / "retrieval.json").read_text(encoding="utf-8"))
    scores = [
        score_model(path, retrieval)
        for path in sorted(RESULTS.glob("model_*.json"))
        if json.loads(path.read_text(encoding="utf-8"))
    ]
    return retrieval, scores
