"""Read-only view over the Week 4 evaluation dataset and results, for the Evaluation Matrix UI.

This module does not run the evaluation - eval/runner.py already does that and takes many
minutes against three models. It exists purely to expose the 27-task dataset (question,
category, expected sections) and the recorded per-model, per-task results (answer, citations,
correctness, retrieval recall/MRR) as read-only API responses, so the marking professor can see
every evaluation question and every model's actual answer without opening a JSON file.

Deliberately does not import eval/dataset.py or eval/metrics.py directly: those are standalone
scripts that assume the working directory is eval/ (relative imports, relative `results/` path),
so importing them from the backend package would be fragile. The 27-task dataset is small and
stable, so it is mirrored here in full; the join/scoring logic against results/*.json is
re-implemented in the ~40 lines below rather than imported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from legalaid.config import RepoRoot, get_settings

EVAL_DIR = RepoRoot / "eval"
RESULTS_DIR = EVAL_DIR / "results"


@dataclass(frozen=True)
class EvalTask:
    id: str
    category: str
    question: str
    expected_sections: list[tuple[str, str]] = field(default_factory=list)
    expected_concepts: list[str] = field(default_factory=list)
    answerable: bool = True
    notes: str = ""


CONTRACT = "Indian Contract Act"
IT = "Information Technology Act"
BSA = "Bharatiya Sakshya Adhiniyam"
CPC = "Code of Civil Procedure"
BNS = "Bharatiya Nyaya Sanhita"

# Mirrors eval/dataset.py TASKS exactly. Kept in sync by hand - the dataset is stable evaluation
# material, not something edited often, and duplicating it here avoids a fragile cross-package
# import of a standalone script directory.
TASKS: list[EvalTask] = [
    EvalTask("L01", "statutory_lookup",
        "What does Section 73 of the Indian Contract Act provide for?",
        [(CONTRACT, "73")], ["compensation", "breach"],
        notes="Baseline. Code Llama answers 'sale of goods' without retrieval."),
    EvalTask("L02", "statutory_lookup",
        "Under the Indian Contract Act, when is an agreement without consideration still valid?",
        [(CONTRACT, "25")], ["writing", "registered", "natural love"]),
    EvalTask("L03", "statutory_lookup",
        "What are the essential requirements for an agreement to be a contract?",
        [(CONTRACT, "10")], ["free consent", "competent", "lawful consideration"]),
    EvalTask("L04", "statutory_lookup",
        "Are electronic records legally recognised in India, and under which provision?",
        [(IT, "4")], ["electronic", "record", "writing"]),
    EvalTask("L05", "statutory_lookup",
        "What does the law say about admissibility of electronic records as evidence?",
        [(BSA, "63"), (BSA, "61")], ["electronic", "admissib"]),
    EvalTask("L06", "statutory_lookup",
        "What is res judicata and which provision governs it?",
        [(CPC, "11")], ["res judicata", "same matter"]),

    EvalTask("R01", "remedy_analysis",
        "My client supplied goods under a written agreement and the buyer refused payment "
        "after delivery. Which provisions support a claim for compensation?",
        [(CONTRACT, "73")], ["compensation", "breach"]),
    EvalTask("R02", "remedy_analysis",
        "Our contract fixes a sum payable on default. Is that clause enforceable and how much "
        "can actually be recovered?",
        [(CONTRACT, "74")], ["reasonable compensation", "penalty"],
        notes="Tests whether the model states the reasonable-compensation cap."),
    EvalTask("R03", "remedy_analysis",
        "The other side made performance impossible after we signed. What is our position?",
        [(CONTRACT, "56")], ["impossible", "void"]),
    EvalTask("R04", "remedy_analysis",
        "We rescinded the contract because the other party breached first. Can we still claim "
        "compensation?",
        [(CONTRACT, "75")], ["rescind", "compensation"]),
    EvalTask("R05", "remedy_analysis",
        "The supplier delivered three months late. Was time of the essence, and what follows?",
        [(CONTRACT, "55")], ["time", "voidable"]),
    EvalTask("R06", "remedy_analysis",
        "The buyer told us in advance they would not perform. What are our options?",
        [(CONTRACT, "39")], ["refus", "perform"]),

    EvalTask("X01", "cross_act",
        "Our entire agreement and all correspondence exist only as email and PDF. Is the "
        "contract valid, and how do we prove these records in court?",
        [(IT, "10A"), (BSA, "63")], ["electronic", "valid", "evidence"],
        notes="Requires combining IT Act validity with BSA admissibility."),
    EvalTask("X02", "cross_act",
        "A buyer took delivery of goods intending never to pay. Is that only a contract breach "
        "or also a criminal offence?",
        [(BNS, "318"), (CONTRACT, "73")], ["cheating", "breach"]),
    EvalTask("X03", "cross_act",
        "We handed over machinery for repair and it was damaged in the workshop. What duty did "
        "they owe and are they liable?",
        [(CONTRACT, "151"), (CONTRACT, "148")], ["bailee", "care"]),
    EvalTask("X04", "cross_act",
        "Someone signed as a surety for our debtor. What is the nature of that obligation and "
        "how does it differ from indemnity?",
        [(CONTRACT, "126"), (CONTRACT, "124")], ["guarantee", "indemnity", "surety"]),

    EvalTask("P01", "procedural",
        "In which court should we institute a suit for breach of a supply contract?",
        [(CPC, "20"), (CPC, "15")], ["cause of action", "jurisdiction"]),
    EvalTask("P02", "procedural",
        "Can we claim interest on the unpaid contract amount in a civil suit?",
        [(CPC, "34")], ["interest"]),
    EvalTask("P03", "procedural",
        "Is a notice required before suing a government department?",
        [(CPC, "80")], ["notice"]),

    EvalTask("D01", "document_analysis",
        "An FIR alleges our client induced delivery of goods worth Rs 18,40,000 and then issued "
        "a cheque that bounced. Which provisions are in play and what is the exposure?",
        [(BNS, "318"), (BNS, "316")], ["cheating", "breach of trust"]),
    EvalTask("D02", "document_analysis",
        "A legal notice claims we breached a supply agreement and demands damages. What "
        "provisions govern the quantum they can claim?",
        [(CONTRACT, "73"), (CONTRACT, "74")], ["compensation", "remote"]),
    EvalTask("D03", "document_analysis",
        "The agreement was signed by someone described as an agent. How do we test whether the "
        "principal is bound?",
        [(CONTRACT, "182")], ["agent", "principal"]),

    EvalTask("C01", "precedent",
        "Are there Supreme Court judgments on arbitration agreements in unstamped contracts?",
        [], ["arbitration"],
        notes="Corpus has the 2023 seven-judge Interplay decision."),
    EvalTask("C02", "precedent",
        "Find Supreme Court authority on delay and damages in construction contracts.",
        [], ["construction"]),

    EvalTask("N01", "negative",
        "What does Section 498A of the Indian Penal Code provide about cruelty by a husband?",
        [], [], answerable=False,
        notes="IPC is not in the corpus (BNS replaced it). Correct answer: insufficient evidence."),
    EvalTask("N02", "negative",
        "What are the GST registration thresholds for an e-commerce operator?",
        [], [], answerable=False,
        notes="No tax legislation ingested at all."),
    EvalTask("N03", "negative",
        "Under the Motor Vehicles Act, what compensation applies in a hit-and-run case?",
        [], [], answerable=False,
        notes="Motor Vehicles Act absent from the corpus."),
]

CATEGORY_LABELS: dict[str, str] = {
    "statutory_lookup": "Statutory lookup",
    "remedy_analysis": "Remedy / scenario analysis",
    "cross_act": "Cross-Act reasoning",
    "procedural": "Procedural",
    "document_analysis": "Document analysis",
    "precedent": "Precedent / case law",
    "negative": "Negative (deliberately unanswerable)",
}

MODEL_FILE_KEYS: dict[str, str] = {
    "llama3.2:3b": "model_llama3.2_3b.json",
    "qwen2.5:3b-instruct": "model_qwen2.5_3b-instruct.json",
    "gemma2:2b": "model_gemma2_2b.json",
}


def _section_key(act: str, number: str) -> str:
    """Normalise so 'Section 73' / 's.73' / '73' in 'Indian Contract Act, 1872' all match."""
    act_token = "".join(ch for ch in act.lower() if ch.isalnum())[:14]
    num = str(number).lower().replace("section", "").replace("s.", "").strip()
    return f"{act_token}:{num}"


def dataset_payload() -> dict[str, Any]:
    """The 27-task dataset with categories, for the Evaluation Matrix page's task list."""
    tasks = [
        {
            "id": t.id,
            "category": t.category,
            "category_label": CATEGORY_LABELS.get(t.category, t.category),
            "question": t.question,
            "expected_sections": [f"{act}, s.{num}" for act, num in t.expected_sections],
            "expected_concepts": t.expected_concepts,
            "answerable": t.answerable,
            "notes": t.notes,
        }
        for t in TASKS
    ]
    by_category: dict[str, int] = {}
    for t in TASKS:
        by_category[t.category] = by_category.get(t.category, 0) + 1
    return {
        "total": len(TASKS),
        "categories": [
            {"key": k, "label": CATEGORY_LABELS.get(k, k), "count": v}
            for k, v in by_category.items()
        ],
        "tasks": tasks,
    }


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def results_payload() -> dict[str, Any]:
    """Every model's recorded answer for every task, joined against expected sections and
    retrieval recall/MRR, so the UI can show question -> category -> per-model answer side by
    side without recomputing anything the eval harness already measured.
    """
    retrieval = _load_json(RESULTS_DIR / "retrieval.json") or {}
    tasks_by_id = {t.id: t for t in TASKS}

    models: dict[str, list[dict[str, Any]]] = {}
    available_models: list[str] = []
    for model_name, filename in MODEL_FILE_KEYS.items():
        records = _load_json(RESULTS_DIR / filename)
        if records is None:
            continue
        available_models.append(model_name)
        per_task: list[dict[str, Any]] = []
        for record in records:
            task = tasks_by_id.get(record.get("task_id"))
            if task is None:
                continue
            expected_keys = {
                _section_key(act, num) for act, num in task.expected_sections
            }
            cited_sections = {
                str(s).strip().lower() for s in (record.get("cited_sections") or [])
            }
            supplied_chunks = set(
                (retrieval.get(task.id) or {}).get("chunk_ids") or []
            )
            cited_chunks = set(record.get("cited_chunk_ids") or [])
            ungrounded = sorted(cited_chunks - supplied_chunks)
            hallucinated = bool(ungrounded)

            if task.answerable:
                hit = cited_sections & expected_keys
                correct = bool(hit) and not hallucinated
            else:
                declined = record.get("evidence_sufficient") is False or not cited_sections
                correct = declined and not hallucinated

            raw_output = record.get("raw_output") or {}
            sections = raw_output.get("sections") or []
            answer_text = " ".join(
                str(s.get("what_it_provides") or "").strip() for s in sections if s.get("what_it_provides")
            ).strip()
            if not answer_text:
                answer_text = "(no answer text recorded)"

            per_task.append({
                "task_id": task.id,
                "category": task.category,
                "question": task.question,
                "answerable": task.answerable,
                "answer_text": answer_text,
                "cited_sections": sorted(cited_sections),
                "expected_sections": sorted(expected_keys),
                "correct": correct,
                "hallucinated": hallucinated,
                "ungrounded_chunk_count": len(ungrounded),
                "evidence_sufficient": record.get("evidence_sufficient"),
                "latency_ms": record.get("latency_ms"),
                "output_tokens": record.get("output_tokens"),
                "parse_ok": record.get("parse_ok"),
                "retrieval_recall": (retrieval.get(task.id) or {}).get("recall"),
                "retrieval_mrr": (retrieval.get(task.id) or {}).get("mrr"),
            })
        models[model_name] = per_task

    return {
        "available_models": available_models,
        "results": models,
    }


def eval_files_present() -> bool:
    """Whether any recorded evaluation results exist, so the UI can show a helpful empty state."""
    return RESULTS_DIR.exists() and any(RESULTS_DIR.glob("model_*.json"))
