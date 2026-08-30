"""Duplicate and near-duplicate detection for the corpus.

Indian legal data duplicates heavily and for structural reasons, not carelessness. The same
judgment appears under several citations; the same central act is adopted verbatim by many
states; a provision quoted inside another provision's illustration reads almost identically to
the original. Left alone, duplicates compete with each other for top-k and quietly reduce the
diversity of retrieved context - the failure is invisible because nothing errors.

Three levels, cheapest first:

1. **Exact** - sha256 over whitespace-normalised text. Catches byte-identical re-ingestion.
2. **Near** - SimHash with Hamming distance. Catches "the same provision with one amended
   word", which an exact hash misses entirely.
3. **Structural** - same act and section number from two different sources.

SimHash is used rather than embedding similarity on purpose: it needs no model, is
deterministic, and runs over the whole corpus in one pass. Embedding-based clustering would be
more semantically aware but would also flag genuinely distinct provisions that happen to
discuss the same subject, which is not duplication.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[a-z0-9]+")

# 64-bit SimHash. A Hamming distance of <= 3 over 64 bits means the documents share almost all
# of their weighted feature mass - in practice, identical text with a few words changed.
SIMHASH_BITS = 64
NEAR_DUPLICATE_DISTANCE = 3


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation that carries no meaning here."""
    return " ".join(_TOKEN.findall((text or "").lower()))


def exact_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def shingles(text: str, size: int = 4) -> list[str]:
    """Overlapping word n-grams. Word-level rather than character-level, because legal text
    differs in whole words when amended, not in characters."""
    tokens = normalise(text).split()
    if len(tokens) < size:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def simhash(text: str, *, shingle_size: int = 4) -> int:
    """64-bit SimHash over word shingles."""
    features = shingles(text, shingle_size)
    if not features:
        return 0
    weights = [0] * SIMHASH_BITS
    for feature in features:
        digest = int(hashlib.md5(feature.encode("utf-8")).hexdigest(), 16)
        for bit in range(SIMHASH_BITS):
            weights[bit] += 1 if (digest >> bit) & 1 else -1
    value = 0
    for bit in range(SIMHASH_BITS):
        if weights[bit] > 0:
            value |= 1 << bit
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class DuplicateGroup:
    kind: str  # "exact" | "near" | "structural"
    members: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "members": self.members,
            "size": len(self.members),
            "detail": self.detail,
        }


@dataclass
class DedupReport:
    total: int = 0
    exact_groups: list[DuplicateGroup] = field(default_factory=list)
    near_groups: list[DuplicateGroup] = field(default_factory=list)
    structural_groups: list[DuplicateGroup] = field(default_factory=list)

    @property
    def exact_duplicates(self) -> int:
        """Redundant copies, not group count: a group of 3 contributes 2 duplicates."""
        return sum(len(g.members) - 1 for g in self.exact_groups)

    @property
    def near_duplicates(self) -> int:
        return sum(len(g.members) - 1 for g in self.near_groups)

    @property
    def structural_duplicates(self) -> int:
        return sum(len(g.members) - 1 for g in self.structural_groups)

    @property
    def duplicate_rate(self) -> float:
        if not self.total:
            return 0.0
        return round((self.exact_duplicates + self.near_duplicates) / self.total, 4)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "structural_duplicates": self.structural_duplicates,
            "duplicate_rate": self.duplicate_rate,
            "exact_groups": [g.to_dict() for g in self.exact_groups[:40]],
            "near_groups": [g.to_dict() for g in self.near_groups[:40]],
            "structural_groups": [g.to_dict() for g in self.structural_groups[:40]],
        }


@dataclass(frozen=True, slots=True)
class DedupItem:
    """One unit of text to check, with enough identity to report a useful group."""

    key: str
    text: str
    act: str = ""
    section: str = ""
    label: str = ""


def find_duplicates(
    items: list[DedupItem],
    *,
    near_distance: int = NEAR_DUPLICATE_DISTANCE,
    min_chars: int = 120,
) -> DedupReport:
    """Group duplicates across three levels.

    Very short texts are excluded from near-duplicate comparison: a 40-character provision like
    "This Act may be called the ..." legitimately resembles the same clause in every other act,
    and reporting those as duplicates would bury the real findings.
    """
    report = DedupReport(total=len(items))

    # ---------------------------------------------------------------- exact
    by_hash: dict[str, list[str]] = defaultdict(list)
    for item in items:
        by_hash[exact_hash(item.text)].append(item.key)
    seen_exact: set[str] = set()
    for digest, keys in by_hash.items():
        if len(keys) > 1:
            report.exact_groups.append(
                DuplicateGroup("exact", sorted(keys), f"identical text, sha256 {digest[:12]}")
            )
            seen_exact.update(keys)

    # ------------------------------------------------------------ structural
    by_section: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in items:
        if item.act and item.section:
            by_section[(item.act.strip().lower(), item.section.strip().lower())].append(item.key)
    for (act, section), keys in by_section.items():
        if len(keys) > 1:
            report.structural_groups.append(
                DuplicateGroup(
                    "structural", sorted(keys), f"same provision cited twice: {act} s.{section}"
                )
            )

    # ------------------------------------------------------------------ near
    # Exact duplicates are excluded so they are not double-counted as near duplicates too.
    candidates = [
        item for item in items if item.key not in seen_exact and len(item.text) >= min_chars
    ]
    signatures = [(item, simhash(item.text)) for item in candidates]

    # Bucket by 16-bit bands so this is not O(n^2) over the whole corpus. Two documents within
    # a small Hamming distance almost always agree on at least one band.
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (_item, signature) in enumerate(signatures):
        for band in range(4):
            buckets[(band, (signature >> (band * 16)) & 0xFFFF)].append(index)

    grouped: dict[int, int] = {}
    groups: dict[int, list[int]] = defaultdict(list)
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        for i, left in enumerate(indices):
            for right in indices[i + 1 :]:
                if hamming(signatures[left][1], signatures[right][1]) > near_distance:
                    continue
                root = grouped.get(left, grouped.get(right, left))
                for member in (left, right):
                    if grouped.get(member) != root:
                        grouped[member] = root
                        if member not in groups[root]:
                            groups[root].append(member)

    for root, members in groups.items():
        if len(members) < 2:
            continue
        keys = sorted(signatures[i][0].key for i in members)
        distance = min(
            hamming(signatures[members[0]][1], signatures[m][1]) for m in members[1:]
        )
        report.near_groups.append(
            DuplicateGroup("near", keys, f"SimHash distance {distance} over {SIMHASH_BITS} bits")
        )

    return report
