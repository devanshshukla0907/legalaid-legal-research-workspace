"""Versioned prompt registry.

Prompts live in `prompts/<name>/<version>.yaml`, never inline in Python. Three reasons this
matters beyond tidiness:

* the UI must report which prompt version produced every model run, so prompts need real
  identity and a changelog;
* prompt text is the main lever for controlling a small model's output, so it should be
  editable without touching code;
* agent behaviour is then reproducible - a trace records the exact version used.

Each file carries its own temperature and token budget, because the agents genuinely differ:
extraction wants determinism, drafting needs a little latitude.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from legalaid.config import get_settings
from legalaid.core.errors import PromptNotFoundError
from legalaid.core.logging import get_logger

logger = get_logger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True, slots=True)
class Prompt:
    """One versioned prompt template."""

    name: str
    version: str
    description: str
    system_prompt: str
    user_prompt: str
    variables: tuple[str, ...]
    temperature: float | None
    max_tokens: int | None
    created_at: str
    path: Path

    def render(self, **values: Any) -> tuple[str, str]:
        """Render (system, user). Missing variables are an error, not a silent blank.

        A silently empty context section is the worst failure mode here: the model would
        answer confidently with no evidence and the trace would look normal.
        """
        missing = [v for v in self.variables if v not in values]
        if missing:
            raise PromptNotFoundError(
                f"Prompt '{self.name}/{self.version}' missing variables: {', '.join(missing)}"
            )
        rendered_user = self.user_prompt
        for key, value in values.items():
            rendered_user = rendered_user.replace(f"{{{key}}}", str(value))

        leftover = _PLACEHOLDER_RE.findall(rendered_user)
        unresolved = [p for p in leftover if p in self.variables]
        if unresolved:
            raise PromptNotFoundError(
                f"Prompt '{self.name}/{self.version}' left placeholders unresolved: {unresolved}"
            )
        return self.system_prompt.strip(), rendered_user.strip()

    @property
    def label(self) -> str:
        return f"{self.name}-{self.version}"


def _prompt_dir() -> Path:
    return get_settings().prompt_dir


def _load_file(path: Path) -> Prompt:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PromptNotFoundError(f"Invalid prompt YAML at {path}: {exc}") from exc

    for required in ("name", "version", "system_prompt", "user_prompt"):
        if not payload.get(required):
            raise PromptNotFoundError(f"Prompt {path} is missing '{required}'")

    return Prompt(
        name=str(payload["name"]),
        version=str(payload["version"]),
        description=str(payload.get("description", "")).strip(),
        system_prompt=str(payload["system_prompt"]),
        user_prompt=str(payload["user_prompt"]),
        variables=tuple(payload.get("variables") or ()),
        temperature=(
            float(payload["temperature"]) if payload.get("temperature") is not None else None
        ),
        max_tokens=(int(payload["max_tokens"]) if payload.get("max_tokens") is not None else None),
        created_at=str(payload.get("created_at", "")),
        path=path,
    )


@lru_cache(maxsize=64)
def load_prompt(name: str, version: str | None = None) -> Prompt:
    """Load a prompt by name, defaulting to the highest available version."""
    directory = _prompt_dir() / name
    if not directory.is_dir():
        raise PromptNotFoundError(
            f"No prompt directory for '{name}' at {directory}. "
            f"Available: {', '.join(sorted(p.name for p in _prompt_dir().iterdir() if p.is_dir()))
                          if _prompt_dir().is_dir() else 'none'}"
        )

    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise PromptNotFoundError(f"No prompt files in {directory}")

    if version:
        # Match on either the filename stem (v1) or the declared version (v1.3).
        for path in files:
            if path.stem == version:
                return _load_file(path)
        for path in files:
            prompt = _load_file(path)
            if prompt.version == version:
                return prompt
        raise PromptNotFoundError(
            f"Prompt '{name}' has no version '{version}'. "
            f"Found: {', '.join(p.stem for p in files)}"
        )

    return _load_file(files[-1])


def all_prompts() -> list[Prompt]:
    """Every prompt on disk, for the Agents page."""
    directory = _prompt_dir()
    if not directory.is_dir():
        return []
    prompts: list[Prompt] = []
    for sub in sorted(directory.iterdir()):
        if not sub.is_dir():
            continue
        for path in sorted(sub.glob("*.yaml")):
            try:
                prompts.append(_load_file(path))
            except PromptNotFoundError as exc:
                logger.warning("prompt_load_failed", path=str(path), error=str(exc))
    return prompts


def reset_cache() -> None:
    load_prompt.cache_clear()
