"""Every agent schema must bound its arrays.

Constrained decoding guarantees the output *shape* but not that a complete object fits the token
budget. Twice now an unbounded array let the model fill its entire budget and get truncated
mid-object at exactly `num_predict`, producing unparseable output after a full minute of
generation. This test makes that class of bug impossible to reintroduce silently.
"""

from __future__ import annotations

from typing import Any

import pytest

from legalaid.agents.prompts import all_prompts
from legalaid.agents.schemas import (
    MAX_SAFE_STRING_LENGTH,
    SCHEMAS,
    required_max_tokens,
    schema_for,
    worst_case_chars,
)


def _nodes_of_type(node: Any, wanted: str, path: str = "") -> list[tuple[str, dict]]:
    """Every subschema of a given `type` in a schema tree, with its path."""
    found: list[tuple[str, dict]] = []
    if isinstance(node, dict):
        if node.get("type") == wanted:
            found.append((path or "<root>", node))
        for key, value in node.items():
            found.extend(_nodes_of_type(value, wanted, f"{path}.{key}" if path else key))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(_nodes_of_type(item, wanted, f"{path}[{index}]"))
    return found


def _arrays(node: Any, path: str = "") -> list[tuple[str, dict]]:
    return _nodes_of_type(node, "array", path)


# Fields that are meant to hold prose and are the deliberate exception to the length rule.
LONG_FORM_FIELDS = {"answer_markdown", "draft_markdown", "factual_summary", "search_query",
                    "reasoning", "retry_guidance", "text"}


class TestSchemaSafety:
    @pytest.mark.parametrize("name", sorted(SCHEMAS))
    def test_every_array_is_bounded(self, name: str) -> None:
        unbounded = [
            path for path, node in _arrays(SCHEMAS[name]) if "maxItems" not in node
        ]
        assert not unbounded, (
            f"schema '{name}' has unbounded arrays at {unbounded}; an unbounded array lets the "
            "model exhaust num_predict and truncate mid-object"
        )

    @pytest.mark.parametrize("name", sorted(SCHEMAS))
    def test_strings_that_summarise_are_length_capped(self, name: str) -> None:
        """Bounding array counts is not enough on its own.

        Observed failure: with `sections` capped at 4 items, the model still exhausted its 620
        output tokens by copying entire statutory passages verbatim into `what_it_provides`,
        dying mid-string on the second section. Fields meant to summarise must be capped;
        genuine prose fields are listed as exceptions.
        """
        unbounded: list[str] = []
        for path, node in _nodes_of_type(SCHEMAS[name], "string"):
            if node.get("enum") or "maxLength" in node:
                continue
            field = path.split(".")[-1].split("[")[0]
            if field in LONG_FORM_FIELDS or field == "items":
                continue
            unbounded.append(path)
        assert not unbounded, (
            f"schema '{name}' has uncapped summary strings at {unbounded}; the model will copy "
            "source text verbatim and exhaust its token budget"
        )

    @pytest.mark.parametrize("name", sorted(SCHEMAS))
    def test_objects_forbid_extra_keys(self, name: str) -> None:
        """Without this the model can invent keys, which is how the statute-echo output slipped through."""
        assert SCHEMAS[name].get("additionalProperties") is False, (
            f"schema '{name}' must set additionalProperties: false"
        )

    @pytest.mark.parametrize("name", sorted(SCHEMAS))
    def test_schema_declares_required_fields(self, name: str) -> None:
        assert SCHEMAS[name].get("required"), f"schema '{name}' declares no required fields"

    def test_every_prompt_has_a_schema(self) -> None:
        """A prompt without a schema falls back to loose JSON mode and can drift off-shape."""
        missing = [p.name for p in all_prompts() if schema_for(p.name) is None]
        assert not missing, f"prompts without a constrained schema: {missing}"

    def test_schema_lookup_is_forgiving(self) -> None:
        assert schema_for("does_not_exist") is None


class TestTokenBudgets:
    """The token budget must be able to hold the schema's worst-case output.

    This is the root cause of three separate truncation failures: `max_tokens` was chosen by
    hand while the schema independently allowed far more, and constrained decoding fills
    whatever the schema permits. Deriving the budget keeps them consistent.
    """

    @pytest.mark.parametrize("name", sorted(SCHEMAS))
    def test_budget_is_not_capped_away(self, name: str) -> None:
        """A schema hitting the ceiling is too loose to be safe.

        The budget is capped at 1500 tokens because ~12 tok/s makes anything larger a
        multi-minute agent. If a schema needs the cap, it can still truncate - so the schema
        must be tightened rather than the cap raised.
        """
        budget = required_max_tokens(name)
        assert budget is not None
        assert budget < 1500, (
            f"schema '{name}' needs the full 1500-token cap; tighten its maxItems/maxLength so "
            "a realistic response fits comfortably"
        )

    @pytest.mark.parametrize("name", sorted(SCHEMAS))
    def test_budget_exceeds_a_realistic_response(self, name: str) -> None:
        """The budget must comfortably hold a realistic (not maximal) response."""
        budget = required_max_tokens(name)
        realistic = worst_case_chars(SCHEMAS[name]) * 0.55 / 3.2
        assert budget is not None and budget >= realistic, (
            f"schema '{name}' realistically emits ~{realistic:.0f} tokens, budget is {budget}"
        )

    def test_unknown_prompt_has_no_budget(self) -> None:
        assert required_max_tokens("does_not_exist") is None


class TestOllamaGrammarLimits:
    """Ollama compiles the schema into a sampling grammar and rejects oversized strings.

    Measured against this Ollama build: `maxLength: 1600` compiles, `2000` returns
    HTTP 400 "failed to parse grammar". That is a request-time failure, so an over-long cap does
    not merely degrade output quality - it disables the agent completely. Both the synthesis and
    drafting agents were silently dead until this was found.
    """

    @pytest.mark.parametrize("name", sorted(SCHEMAS))
    def test_no_string_exceeds_the_grammar_limit(self, name: str) -> None:
        oversized = [
            (path, node["maxLength"])
            for path, node in _nodes_of_type(SCHEMAS[name], "string")
            if node.get("maxLength", 0) > MAX_SAFE_STRING_LENGTH
        ]
        assert not oversized, (
            f"schema '{name}' has maxLength above {MAX_SAFE_STRING_LENGTH} at {oversized}; "
            "Ollama will reject the request with HTTP 400 'failed to parse grammar'"
        )
