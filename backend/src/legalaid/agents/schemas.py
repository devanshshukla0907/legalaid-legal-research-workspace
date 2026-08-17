"""JSON Schemas used to constrain agent output at the decoder.

Ollama accepts a JSON Schema as its `format` parameter and constrains token sampling to it, so
output cannot be malformed or off-schema. That is not a nicety here - it is what makes a 7B
code-specialised model usable for structured legal output at all.

Measured before constrained decoding: asked politely for a schema, Code Llama ignored it and
echoed the retrieved statute's Illustrations back as ``{"73": [{"a": "A contracts to sell..."}]}``.
Both research agents failed to parse, twice each, costing ~10 minutes. The schemas below remove
that failure class entirely.

Kept deliberately shallow and small. Every optional field and every nested level is another
token the model must spend, and at ~12 tok/s tokens are the binding constraint.
"""

from __future__ import annotations

from typing import Any

# `additionalProperties: false` is essential: without it the model can bolt on the extra keys
# it invented before, which is how the statute-echo output slipped through.

LEGAL_RESEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "legal_issues": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 160},
        },
        # maxItems is load-bearing, not cosmetic. Constrained decoding guarantees the *shape*
        # of the output but not that it fits the token budget: without a cap the model opened a
        # long "sections" array and was truncated at exactly num_predict, producing valid-so-far
        # but incomplete JSON. Bounding the arrays keeps a complete object inside the budget.
        # `maxLength` matters as much as `maxItems`. Bounding only the array count did not stop
        # truncation: the model copied whole statutory passages verbatim into
        # `what_it_provides` and `relevance`, spending all 620 output tokens on two sections and
        # dying mid-string. Capping the strings forces a summary instead of a transcription -
        # which is what the field is for, since the full text is one click away in the chunk.
        "sections": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "maxLength": 60},
                    "act": {"type": "string", "maxLength": 90},
                    "chunk_id": {"type": "string", "maxLength": 48},
                    "what_it_provides": {"type": "string", "maxLength": 220},
                    "relevance": {"type": "string", "maxLength": 220},
                },
                "required": ["section", "act", "chunk_id", "what_it_provides", "relevance"],
                "additionalProperties": False,
            },
        },
        "reasoning": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 400},
        },
        "uncertainties": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 240},
        },
        "evidence_sufficient": {"type": "boolean"},
    },
    "required": ["legal_issues", "sections", "reasoning", "evidence_sufficient"],
    "additionalProperties": False,
}

SIMILAR_CASES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cases": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "case_title": {"type": "string", "maxLength": 200},
                    "citation": {"type": "string", "maxLength": 80},
                    "chunk_id": {"type": "string", "maxLength": 48},
                    "why_similar": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {"type": "string", "maxLength": 200},
                    },
                    "relevant_principle": {"type": "string", "maxLength": 300},
                    "outcome": {"type": "string", "maxLength": 120},
                    "distinguishing_factors": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {"type": "string", "maxLength": 200},
                    },
                    "strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                },
                "required": [
                    "case_title",
                    "citation",
                    "chunk_id",
                    "why_similar",
                    "relevant_principle",
                    "strength",
                ],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string", "maxLength": 300},
        "evidence_sufficient": {"type": "boolean"},
    },
    "required": ["cases", "note", "evidence_sufficient"],
    "additionalProperties": False,
}

DOCUMENT_UNDERSTANDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # Every array is bounded. Unbounded, the model filled its whole token budget listing
        # parties and dates and was cut off mid-object at exactly num_predict - the same
        # truncation failure seen earlier with `sections`.
        "parties": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 110},
                    "role": {"type": "string", "maxLength": 40},
                    "chunk_id": {"type": "string", "maxLength": 48},
                },
                "required": ["name", "role", "chunk_id"],
                "additionalProperties": False,
            },
        },
        # 8 dates was too many: the model listed every date in the document with the event
        # field set to the literal string "Date", filling the budget with nothing useful.
        "key_dates": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "maxLength": 32},
                    "event": {"type": "string", "maxLength": 110},
                    "chunk_id": {"type": "string", "maxLength": 48},
                },
                "required": ["date", "event", "chunk_id"],
                "additionalProperties": False,
            },
        },
        "amounts": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "amount": {"type": "string", "maxLength": 40},
                    "what_for": {"type": "string", "maxLength": 110},
                    "chunk_id": {"type": "string", "maxLength": 48},
                },
                "required": ["amount", "what_for", "chunk_id"],
                "additionalProperties": False,
            },
        },
        "obligations": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "who": {"type": "string", "maxLength": 90},
                    "what": {"type": "string", "maxLength": 160},
                    "chunk_id": {"type": "string", "maxLength": 48},
                },
                "required": ["who", "what", "chunk_id"],
                "additionalProperties": False,
            },
        },
        "factual_summary": {"type": "string", "maxLength": 380},
        "potential_issues": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 140},
        },
    },
    "required": ["factual_summary", "potential_issues"],
    "additionalProperties": False,
}

DRAFTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 120},
        # The draft is the deliverable, so it gets the largest allowance of any field - but a
        # bounded one. ~2600 chars is a substantial first draft and still leaves the agent
        # inside a workable latency at ~12 tok/s.
        "draft_markdown": {"type": "string", "maxLength": 1400},
        "authorities_relied_on": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string", "maxLength": 110},
        },
        "missing_information": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 110},
        },
        "assumptions": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 140},
        },
        "review_points": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 140},
        },
    },
    "required": ["title", "draft_markdown", "authorities_relied_on", "missing_information"],
    "additionalProperties": False,
}

VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "revise", "fail"]},
        # Citations are ids, so they are short; the deterministic audit finds these anyway, so
        # the LLM reviewer only needs to report a handful.
        "unsupported_citations": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string", "maxLength": 60},
        },
        "unsupported_claims": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 200},
        },
        "overconfident_statements": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 200},
        },
        "retry_guidance": {"type": "string", "maxLength": 600},
        "confidence": {"type": "number"},
    },
    "required": ["verdict", "unsupported_citations", "retry_guidance"],
    "additionalProperties": False,
}

INTAKE_ROUTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "input_kind": {
            "type": "string",
            "enum": ["question", "document", "question_with_document", "unclear"],
        },
        "document_type": {
            "type": "string",
            "enum": [
                "fir",
                "legal_notice",
                "agreement",
                "complaint",
                "reply",
                "court_order",
                "client_statement",
                "judgment",
                "other",
                "none",
            ],
        },
        "task": {
            "type": "string",
            "enum": [
                "legal_research",
                "find_precedents",
                "draft_document",
                "analyse_document",
                "full_research",
            ],
        },
        "needs_law": {"type": "boolean"},
        "needs_cases": {"type": "boolean"},
        "needs_drafting": {"type": "boolean"},
        "needs_document_facts": {"type": "boolean"},
        "is_legal_matter": {"type": "boolean"},
        # 25-60 words of dense legal language, per the prompt.
        "search_query": {"type": "string", "maxLength": 600},
        "legal_concepts": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 80},
        },
        "suggested_draft_type": {
            "type": "string",
            "enum": [
                "notice_reply",
                "petitioner_arguments",
                "written_submission",
                "case_summary",
                "research_questions",
                "none",
            ],
        },
        "reasoning": {"type": "string", "maxLength": 400},
    },
    "required": [
        "input_kind",
        "task",
        "needs_law",
        "needs_cases",
        "needs_drafting",
        "needs_document_facts",
        "is_legal_matter",
        "search_query",
        "reasoning",
    ],
    "additionalProperties": False,
}

SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # The prose answer is the main deliverable; ~2000 chars is 4-5 solid paragraphs.
        "answer_markdown": {"type": "string", "maxLength": 1400},
        "key_points": {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "maxLength": 180},
        },
        "inference": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 220},
        },
        "uncertainties": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 160},
        },
        "next_steps": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 160},
        },
        "authorities_cited": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string", "maxLength": 110},
        },
        "confidence": {"type": "string", "enum": ["high", "moderate", "low"]},
    },
    "required": ["answer_markdown", "key_points", "confidence"],
    "additionalProperties": False,
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "intake_router": INTAKE_ROUTER_SCHEMA,
    "synthesis": SYNTHESIS_SCHEMA,
    "legal_research": LEGAL_RESEARCH_SCHEMA,
    "similar_cases": SIMILAR_CASES_SCHEMA,
    "document_understanding": DOCUMENT_UNDERSTANDING_SCHEMA,
    "drafting": DRAFTING_SCHEMA,
    "verification": VERIFICATION_SCHEMA,
}


def schema_for(prompt_name: str) -> dict[str, Any] | None:
    return SCHEMAS.get(prompt_name)


# Rough characters-per-token for this JSON-heavy output. Deliberately conservative (real ratio
# is nearer 4) so the derived budget errs on the generous side.
_CHARS_PER_TOKEN = 3.2

# Ollama compiles the JSON Schema into a sampling grammar, and that compiler fails outright on
# large `maxLength` values: measured on this build, 1600 compiles and 2000 returns
# HTTP 400 "failed to parse grammar". The failure is at request time, so an over-long cap does
# not degrade output - it disables the agent entirely. 1400 leaves margin below the observed
# limit. `test_schemas.py` enforces it.
MAX_SAFE_STRING_LENGTH = 1400


def worst_case_chars(node: Any) -> int:
    """Largest JSON document the schema permits, in characters."""
    kind = node.get("type") if isinstance(node, dict) else None
    if kind == "object":
        properties = node.get("properties") or {}
        # key + quotes + colon + comma overhead per property.
        return sum(worst_case_chars(v) + len(k) + 6 for k, v in properties.items()) + 4
    if kind == "array":
        item = node.get("items") or {}
        return node.get("maxItems", 1) * (worst_case_chars(item) + 4) + 4
    if kind == "string":
        if node.get("enum"):
            return max(len(str(v)) for v in node["enum"]) + 2
        # An uncapped string is the bug this module exists to prevent; assume it is large.
        return node.get("maxLength", 1200) + 2
    if kind in {"number", "integer"}:
        return 12
    if kind == "boolean":
        return 6
    return 16


# Fields never all hit their maximum at once - a real answer fills perhaps half of what the
# schema permits. Budgeting for the absolute worst case would mean 3000+ tokens per agent, which
# at ~12 tok/s is four minutes each. This fraction is the realistic ceiling, and `SAFETY_MARGIN`
# keeps headroom so a verbose-but-valid response still completes.
_REALISTIC_FRACTION = 0.55
_SAFETY_MARGIN = 1.25


def required_max_tokens(prompt_name: str, *, floor: int = 320, ceiling: int = 1500) -> int | None:
    """Token budget sized from the schema rather than guessed.

    Hand-picking `max_tokens` failed three times: constrained decoding fills whatever the schema
    permits, so any budget below the schema's realistic output guarantees truncation mid-object.
    Deriving it keeps schema and budget consistent - tighten the schema and the budget follows.

    The budget is capped: if a schema needs more than `ceiling` the schema itself is too loose,
    and `test_schemas.py` fails rather than letting a four-minute agent through silently.
    """
    schema = SCHEMAS.get(prompt_name)
    if schema is None:
        return None
    realistic = worst_case_chars(schema) * _REALISTIC_FRACTION / _CHARS_PER_TOKEN
    return min(ceiling, max(floor, int(realistic * _SAFETY_MARGIN)))
