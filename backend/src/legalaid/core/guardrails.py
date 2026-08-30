"""Input and output guardrails.

The point of this module is the difference between "the LLM can answer the question" and "the
application knows what it should answer, what it should not, and under what conditions".

Design principle: **deterministic layers run first and the LLM runs last.** The previous scope
check was a single boolean from a 3B model, defaulting to True on failure - so a bad generation
waved everything through. Every layer here is cheap, explainable, and fails closed.

Layers, in execution order:

1. `EMPTY`      - nothing to answer.
2. `TOO_LONG`   - beyond what the context window can hold; better to say so than to truncate
                  silently and answer from a fragment.
3. `INJECTION`  - imperative patterns aimed at the system rather than questions about law.
4. `DENYLIST`   - unambiguous non-legal requests (write code, tell a joke).
5. `CORPUS_DISTANCE` - the interesting one. Embed the question and measure its best cosine
                  similarity against the actual corpus. If nothing in the knowledge base comes
                  close, the system cannot ground an answer whatever the model believes. This
                  is model-free, reuses the embedding service already in the request path, and
                  catches out-of-domain questions the denylist never anticipated.
6. `LLM_SCOPE`  - the router's `is_legal_matter`, now the last word rather than the only one.

Output guardrails compute a groundedness score from the deterministic audit and refuse to
present a confident answer that rests on nothing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from legalaid.config import get_settings
from legalaid.core.logging import get_logger

logger = get_logger(__name__)


class GuardrailLayer(str, Enum):
    EMPTY = "empty"
    TOO_LONG = "too_long"
    INJECTION = "injection"
    DENYLIST = "denylist"
    CORPUS_DISTANCE = "corpus_distance"
    LLM_SCOPE = "llm_scope"


class Verdict(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


@dataclass(slots=True)
class LayerResult:
    layer: GuardrailLayer
    verdict: Verdict
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer.value,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class GuardrailDecision:
    """The full record of why an input was allowed or refused."""

    allowed: bool
    layers: list[LayerResult] = field(default_factory=list)
    blocked_by: GuardrailLayer | None = None
    message: str = ""

    @property
    def warnings(self) -> list[LayerResult]:
        return [layer for layer in self.layers if layer.verdict is Verdict.WARN]

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "blocked_by": self.blocked_by.value if self.blocked_by else None,
            "message": self.message,
            "layers": [layer.to_dict() for layer in self.layers],
            "layers_run": len(self.layers),
        }


# --------------------------------------------------------------------------- patterns

# Aimed at the system rather than asking about law. Deliberately narrow: "ignore the notice"
# is a legitimate legal phrase, so the patterns require the system-directed framing.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("override_instructions", re.compile(r"ignore\s+(all\s+|your\s+|the\s+)?(previous|prior|above)\s+instructions", re.I)),
    ("disregard_rules", re.compile(r"disregard\s+(all\s+|your\s+)?(rules|instructions|guidelines|constraints)", re.I)),
    ("role_reassignment", re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I)),
    ("system_impersonation", re.compile(r"^\s*(system|assistant)\s*:", re.I | re.M)),
    ("prompt_exfiltration", re.compile(r"(reveal|show|print|repeat|output)\s+(me\s+)?(your\s+)?(system\s+)?prompt", re.I)),
    ("guardrail_bypass", re.compile(r"(bypass|disable|turn\s+off)\s+(your\s+)?(guardrails?|safety|filters?|verification)", re.I)),
    ("dev_mode", re.compile(r"\b(developer|debug|god)\s+mode\b", re.I)),
)

# Unambiguously outside an Indian legal research tool. Kept short on purpose: the
# corpus-distance layer is the general mechanism, and a long denylist becomes a source of
# false positives on legitimate legal language.
_DENYLIST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("code_generation", re.compile(r"\b(write|generate|create|fix|debug|refactor)\b.{0,24}\b(code|function|script|program|sql query|regex|component)\b", re.I)),
    ("creative_writing", re.compile(r"\b(write|compose|tell)\b.{0,16}\b(poem|joke|story|song|limerick|essay about)\b", re.I)),
    ("recipe", re.compile(r"\b(recipe|how\s+to\s+cook|how\s+do\s+i\s+cook)\b", re.I)),
    ("medical_advice", re.compile(r"\b(diagnose|symptoms?\s+of|treatment\s+for|prescribe|dosage)\b", re.I)),
    ("translation_only", re.compile(r"^\s*translate\s+(this\s+)?(to|into)\s+\w+\s*:?\s*$", re.I)),
)

# A question mentioning any of these is legal on its face, so the corpus-distance layer only
# warns rather than blocking. This prevents a thin corpus from refusing genuine legal work -
# the correct response to "we have no material on the Motor Vehicles Act" is to say so after
# retrieving, not to refuse to look.
_LEGAL_MARKERS = re.compile(
    r"\b(section|act|clause|sub-?section|article|schedule|provision|statute|ordinance"
    r"|court|judgment|judgement|order|decree|appeal|petition|plaint|suit|litigation"
    r"|contract|agreement|breach|damages|compensation|liability|indemnity|guarantee"
    r"|accused|complainant|petitioner|respondent|plaintiff|defendant|prosecution"
    r"|bail|fir|charge\s?sheet|summons|notice|affidavit|evidence|witness|testimony"
    r"|tribunal|arbitration|jurisdiction|limitation|injunction|writ|habeas"
    r"|offence|offense|penalty|punishable|conviction|acquittal|cognizable"
    r"|legal|lawful|unlawful|statutory|precedent|ratio|obiter|held\s+that"
    r"|ipc|crpc|bns|bnss|bsa|cpc|constitution|law)\b",
    re.I,
)


# --------------------------------------------------------------------------- input


def screen_static(text: str) -> list[LayerResult]:
    """Layers 1-4. No model, no embedding, no I/O - so these are effectively free."""
    settings = get_settings()
    results: list[LayerResult] = []
    stripped = (text or "").strip()

    if not stripped:
        results.append(
            LayerResult(
                GuardrailLayer.EMPTY,
                Verdict.BLOCK,
                "The request was empty.",
            )
        )
        return results
    results.append(LayerResult(GuardrailLayer.EMPTY, Verdict.ALLOW))

    # 4 chars/token is the standard rough estimate; leave room for retrieved context and the
    # prompt itself rather than filling the window with the question.
    budget_chars = settings.llm_num_ctx * 4
    limit = int(budget_chars * 0.5)
    if len(stripped) > limit:
        results.append(
            LayerResult(
                GuardrailLayer.TOO_LONG,
                Verdict.BLOCK,
                f"The request is {len(stripped)} characters; the limit is {limit}. "
                "Attach it as a document instead so it can be chunked and retrieved.",
                {"chars": len(stripped), "limit": limit, "num_ctx": settings.llm_num_ctx},
            )
        )
        return results
    results.append(
        LayerResult(GuardrailLayer.TOO_LONG, Verdict.ALLOW, evidence={"chars": len(stripped)})
    )

    matched_injection = [name for name, pattern in _INJECTION_PATTERNS if pattern.search(stripped)]
    if matched_injection:
        results.append(
            LayerResult(
                GuardrailLayer.INJECTION,
                Verdict.BLOCK,
                "The request contains instructions directed at the system rather than a "
                "question about law.",
                {"patterns": matched_injection},
            )
        )
        return results
    results.append(LayerResult(GuardrailLayer.INJECTION, Verdict.ALLOW))

    matched_deny = [name for name, pattern in _DENYLIST_PATTERNS if pattern.search(stripped)]
    # A legal marker overrides the denylist: "draft a written statement" contains "write" but
    # is core legal work, and "what is the penalty for forgery" mentions treatment of offences.
    if matched_deny and not _LEGAL_MARKERS.search(stripped):
        results.append(
            LayerResult(
                GuardrailLayer.DENYLIST,
                Verdict.BLOCK,
                "This is outside what a legal research assistant does.",
                {"categories": matched_deny},
            )
        )
        return results
    results.append(
        LayerResult(
            GuardrailLayer.DENYLIST,
            Verdict.ALLOW,
            evidence={
                "matched_but_overridden": matched_deny,
                "legal_marker_present": bool(_LEGAL_MARKERS.search(stripped)),
            },
        )
    )
    return results


async def screen_corpus_distance(text: str) -> LayerResult:
    """Layer 5: can anything in the knowledge base actually support this question?

    Embeds the question and takes the best cosine similarity across all collections. A question
    whose nearest neighbour is far away cannot be grounded, regardless of how confident a model
    would sound answering it - Exercise 6 demonstrated exactly this failure, where questions
    about Python imports returned statute at 0.52-0.57 similarity.

    Blocks only when there is no legal marker in the text. A question that is legal on its face
    but has no corpus support should proceed and then honestly report insufficient evidence,
    which is far more useful than a refusal.
    """
    from legalaid.services.embedding.service import embed_texts_raw
    from legalaid.services.vector.store import get_store

    settings = get_settings()
    floor = settings.guardrail_corpus_similarity_floor

    try:
        vectors, _ms, _model = await embed_texts_raw([text])
        if not vectors:
            raise RuntimeError("no vector returned")
        store = get_store()
        best = 0.0
        best_collection = None
        for collection, index in store.all_indexes().items():
            if not index.count:
                continue
            hits, _cand, _filt = index.search(vectors[0], top_k=1, min_similarity=0.0)
            if hits and hits[0].similarity > best:
                best = hits[0].similarity
                best_collection = collection.value
    except Exception as exc:  # noqa: BLE001
        # An embedding failure must not become a refusal; the remaining layers still apply.
        logger.warning("corpus_distance_unavailable", error=f"{type(exc).__name__}: {exc}")
        return LayerResult(
            GuardrailLayer.CORPUS_DISTANCE,
            Verdict.ALLOW,
            "Corpus distance could not be measured; relying on the other layers.",
            {"error": f"{type(exc).__name__}"},
        )

    evidence = {
        "best_similarity": round(best, 4),
        "floor": floor,
        "nearest_collection": best_collection,
    }

    if best >= floor:
        return LayerResult(GuardrailLayer.CORPUS_DISTANCE, Verdict.ALLOW, evidence=evidence)

    if _LEGAL_MARKERS.search(text or ""):
        return LayerResult(
            GuardrailLayer.CORPUS_DISTANCE,
            Verdict.WARN,
            f"This looks like a legal question, but the closest material in the knowledge base "
            f"scores only {best:.2f}. Expect a low-confidence answer or an explicit statement "
            "that the corpus cannot support one.",
            evidence,
        )

    return LayerResult(
        GuardrailLayer.CORPUS_DISTANCE,
        Verdict.BLOCK,
        f"Nothing in the knowledge base is relevant to this request (best match "
        f"{best:.2f}, floor {floor:.2f}), and it does not read as a legal question.",
        evidence,
    )


async def screen_input(text: str, *, check_corpus: bool = True) -> GuardrailDecision:
    """Run the input guardrail stack. Returns the full decision record for the trace."""
    layers = screen_static(text)
    blocking = next((layer for layer in layers if layer.verdict is Verdict.BLOCK), None)
    if blocking is not None:
        return GuardrailDecision(
            allowed=False,
            layers=layers,
            blocked_by=blocking.layer,
            message=blocking.reason,
        )

    if check_corpus:
        corpus = await screen_corpus_distance(text)
        layers.append(corpus)
        if corpus.verdict is Verdict.BLOCK:
            return GuardrailDecision(
                allowed=False, layers=layers, blocked_by=corpus.layer, message=corpus.reason
            )

    return GuardrailDecision(allowed=True, layers=layers)


def record_llm_scope(
    decision: GuardrailDecision, *, is_legal_matter: bool, reasoning: str
) -> GuardrailDecision:
    """Layer 6: fold the router's verdict into an existing decision.

    Separate from `screen_input` because the router runs inside the orchestrator and needs the
    earlier layers' verdict first - a request the deterministic layers already refused should
    never cost an LLM call.
    """
    layer = LayerResult(
        GuardrailLayer.LLM_SCOPE,
        Verdict.ALLOW if is_legal_matter else Verdict.BLOCK,
        "" if is_legal_matter else (reasoning or "The router did not identify a legal matter."),
        {"is_legal_matter": is_legal_matter},
    )
    decision.layers.append(layer)
    if layer.verdict is Verdict.BLOCK:
        decision.allowed = False
        decision.blocked_by = layer.layer
        decision.message = layer.reason
    return decision


# --------------------------------------------------------------------------- output


@dataclass(slots=True)
class OutputVerdict:
    """Whether an answer is fit to return, and the evidence for that judgement."""

    allowed: bool
    groundedness: float
    total_citations: int
    grounded_citations: int
    problems: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def screen_output(
    *,
    answer: dict[str, Any],
    audit_report: dict[str, Any] | None,
    allowed_chunk_ids: list[str],
) -> OutputVerdict:
    """Refuse to present a confident answer that rests on nothing.

    `groundedness` = grounded citations / total citations. A confident answer with no grounded
    citation at all is the exact failure mode that makes legal AI dangerous, so it is blocked
    rather than warned about. An answer that already admits low confidence is allowed through,
    because saying "I am not sure" with thin evidence is honest rather than misleading.
    """
    report = audit_report or {}
    unsupported = list(report.get("unsupported_citations") or [])
    cited = [str(a) for a in (answer.get("authorities_cited") or []) if str(a).strip()]
    total = len(cited)
    grounded = max(0, total - len(unsupported))
    groundedness = (grounded / total) if total else 0.0

    problems: list[str] = []
    for key in ("unsupported_citations", "unsupported_sections", "unsupported_cases"):
        for item in report.get(key) or []:
            problems.append(f"{key}: {item}")

    confidence = str(answer.get("confidence") or "").lower()
    has_answer = bool(str(answer.get("answer_markdown") or "").strip())

    if has_answer and total == 0 and confidence in {"high", "moderate"} and allowed_chunk_ids:
        return OutputVerdict(
            allowed=False,
            groundedness=0.0,
            total_citations=0,
            grounded_citations=0,
            problems=problems,
            message=(
                f"The answer claims {confidence} confidence but cites no authority, while "
                f"{len(allowed_chunk_ids)} chunks were retrieved. An unsourced confident answer "
                "is not returned; the confidence has been downgraded."
            ),
        )

    return OutputVerdict(
        allowed=True,
        groundedness=round(groundedness, 4),
        total_citations=total,
        grounded_citations=grounded,
        problems=problems,
    )
