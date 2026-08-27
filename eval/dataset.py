"""Evaluation dataset: 27 tasks representing what a lawyer actually asks LegalAId.

Every `expected_sections` entry was verified to exist in the live knowledge base
(scripts/list_sections.py), so a model that fails is failing at reasoning or grounding, not
being marked down for a gap in the corpus.

The `NEGATIVE` category is deliberate and load-bearing: those subjects are genuinely absent
from the corpus, so the correct behaviour is to report insufficient evidence. Without such
tasks a hallucination rate cannot be measured - a model that always answers confidently would
score identically to one that knows when to stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalTask:
    id: str
    category: str
    question: str
    # Sections the answer should rely on, as (act_fragment, section_number).
    expected_sections: list[tuple[str, str]] = field(default_factory=list)
    # Substrings that should appear somewhere in the answer (concept coverage).
    expected_concepts: list[str] = field(default_factory=list)
    # True when the corpus cannot support an answer; the model should say so.
    answerable: bool = True
    notes: str = ""


CONTRACT = "Indian Contract Act"
IT = "Information Technology Act"
BSA = "Bharatiya Sakshya Adhiniyam"
CPC = "Code of Civil Procedure"
BNS = "Bharatiya Nyaya Sanhita"


TASKS: list[EvalTask] = [
    # ---------------------------------------------------------- statutory lookup (6)
    EvalTask(
        "L01", "statutory_lookup",
        "What does Section 73 of the Indian Contract Act provide for?",
        [(CONTRACT, "73")],
        ["compensation", "breach"],
        notes="Baseline. Code Llama answers 'sale of goods' without retrieval.",
    ),
    EvalTask(
        "L02", "statutory_lookup",
        "Under the Indian Contract Act, when is an agreement without consideration still valid?",
        [(CONTRACT, "25")],
        ["writing", "registered", "natural love"],
    ),
    EvalTask(
        "L03", "statutory_lookup",
        "What are the essential requirements for an agreement to be a contract?",
        [(CONTRACT, "10")],
        ["free consent", "competent", "lawful consideration"],
    ),
    EvalTask(
        "L04", "statutory_lookup",
        "Are electronic records legally recognised in India, and under which provision?",
        [(IT, "4")],
        ["electronic", "record", "writing"],
    ),
    EvalTask(
        "L05", "statutory_lookup",
        "What does the law say about admissibility of electronic records as evidence?",
        [(BSA, "63"), (BSA, "61")],
        ["electronic", "admissib"],
    ),
    EvalTask(
        "L06", "statutory_lookup",
        "What is res judicata and which provision governs it?",
        [(CPC, "11")],
        ["res judicata", "same matter"],
    ),

    # ---------------------------------------------------- remedy / scenario analysis (6)
    EvalTask(
        "R01", "remedy_analysis",
        "My client supplied goods under a written agreement and the buyer refused payment "
        "after delivery. Which provisions support a claim for compensation?",
        [(CONTRACT, "73")],
        ["compensation", "breach"],
    ),
    EvalTask(
        "R02", "remedy_analysis",
        "Our contract fixes a sum payable on default. Is that clause enforceable and how much "
        "can actually be recovered?",
        [(CONTRACT, "74")],
        ["reasonable compensation", "penalty"],
        notes="Tests whether the model states the reasonable-compensation cap.",
    ),
    EvalTask(
        "R03", "remedy_analysis",
        "The other side made performance impossible after we signed. What is our position?",
        [(CONTRACT, "56")],
        ["impossible", "void"],
    ),
    EvalTask(
        "R04", "remedy_analysis",
        "We rescinded the contract because the other party breached first. Can we still claim "
        "compensation?",
        [(CONTRACT, "75")],
        ["rescind", "compensation"],
    ),
    EvalTask(
        "R05", "remedy_analysis",
        "The supplier delivered three months late. Was time of the essence, and what follows?",
        [(CONTRACT, "55")],
        ["time", "voidable"],
    ),
    EvalTask(
        "R06", "remedy_analysis",
        "The buyer told us in advance they would not perform. What are our options?",
        [(CONTRACT, "39")],
        ["refus", "perform"],
    ),

    # ------------------------------------------------------------ cross-act reasoning (4)
    EvalTask(
        "X01", "cross_act",
        "Our entire agreement and all correspondence exist only as email and PDF. Is the "
        "contract valid, and how do we prove these records in court?",
        [(IT, "10A"), (BSA, "63")],
        ["electronic", "valid", "evidence"],
        notes="Requires combining IT Act validity with BSA admissibility.",
    ),
    EvalTask(
        "X02", "cross_act",
        "A buyer took delivery of goods intending never to pay. Is that only a contract breach "
        "or also a criminal offence?",
        [(BNS, "318"), (CONTRACT, "73")],
        ["cheating", "breach"],
    ),
    EvalTask(
        "X03", "cross_act",
        "We handed over machinery for repair and it was damaged in the workshop. What duty did "
        "they owe and are they liable?",
        [(CONTRACT, "151"), (CONTRACT, "148")],
        ["bailee", "care"],
    ),
    EvalTask(
        "X04", "cross_act",
        "Someone signed as a surety for our debtor. What is the nature of that obligation and "
        "how does it differ from indemnity?",
        [(CONTRACT, "126"), (CONTRACT, "124")],
        ["guarantee", "indemnity", "surety"],
    ),

    # ----------------------------------------------------------- procedural (3)
    EvalTask(
        "P01", "procedural",
        "In which court should we institute a suit for breach of a supply contract?",
        [(CPC, "20"), (CPC, "15")],
        ["cause of action", "jurisdiction"],
    ),
    EvalTask(
        "P02", "procedural",
        "Can we claim interest on the unpaid contract amount in a civil suit?",
        [(CPC, "34")],
        ["interest"],
    ),
    EvalTask(
        "P03", "procedural",
        "Is a notice required before suing a government department?",
        [(CPC, "80")],
        ["notice"],
    ),

    # --------------------------------------------------------- document analysis (3)
    EvalTask(
        "D01", "document_analysis",
        "An FIR alleges our client induced delivery of goods worth Rs 18,40,000 and then issued "
        "a cheque that bounced. Which provisions are in play and what is the exposure?",
        [(BNS, "318"), (BNS, "316")],
        ["cheating", "breach of trust"],
    ),
    EvalTask(
        "D02", "document_analysis",
        "A legal notice claims we breached a supply agreement and demands damages. What "
        "provisions govern the quantum they can claim?",
        [(CONTRACT, "73"), (CONTRACT, "74")],
        ["compensation", "remote"],
    ),
    EvalTask(
        "D03", "document_analysis",
        "The agreement was signed by someone described as an agent. How do we test whether the "
        "principal is bound?",
        [(CONTRACT, "182")],
        ["agent", "principal"],
    ),

    # ------------------------------------------------------------- precedent (2)
    EvalTask(
        "C01", "precedent",
        "Are there Supreme Court judgments on arbitration agreements in unstamped contracts?",
        [],
        ["arbitration"],
        notes="Corpus has the 2023 seven-judge Interplay decision.",
    ),
    EvalTask(
        "C02", "precedent",
        "Find Supreme Court authority on delay and damages in construction contracts.",
        [],
        ["construction"],
    ),

    # ------------------------------------------------------- negative / adversarial (3)
    EvalTask(
        "N01", "negative",
        "What does Section 498A of the Indian Penal Code provide about cruelty by a husband?",
        [],
        [],
        answerable=False,
        notes="IPC is not in the corpus (BNS replaced it). Correct answer: insufficient evidence.",
    ),
    EvalTask(
        "N02", "negative",
        "What are the GST registration thresholds for an e-commerce operator?",
        [],
        [],
        answerable=False,
        notes="No tax legislation ingested at all.",
    ),
    EvalTask(
        "N03", "negative",
        "Under the Motor Vehicles Act, what compensation applies in a hit-and-run case?",
        [],
        [],
        answerable=False,
        notes="Motor Vehicles Act absent from the corpus.",
    ),
]


def by_category() -> dict[str, list[EvalTask]]:
    out: dict[str, list[EvalTask]] = {}
    for task in TASKS:
        out.setdefault(task.category, []).append(task)
    return out


if __name__ == "__main__":
    print(f"{len(TASKS)} tasks\n")
    for name, items in by_category().items():
        answerable = sum(1 for t in items if t.answerable)
        print(f"  {name:<20} {len(items):>2} tasks  ({answerable} answerable)")
    print(f"\nanswerable    : {sum(1 for t in TASKS if t.answerable)}")
    print(f"unanswerable  : {sum(1 for t in TASKS if not t.answerable)}")
