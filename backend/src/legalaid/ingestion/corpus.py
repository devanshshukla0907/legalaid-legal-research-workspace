"""The MVP legal corpus manifest.

Every entry's `handle` was verified against the live India Code site: the title and section
count below were read back from the act page, not assumed. `scripts/find_handles.py` reproduces
that verification.

The corpus is deliberately small. The brief targets 5-10 acts and 100-300 provisions, and
quality plus traceability matter more than volume. `section_filter` keeps high-value provisions
for large codes so the index stays focused and embedding time stays reasonable on this hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from legalaid.domain.enums import Collection


@dataclass(frozen=True, slots=True)
class ActSpec:
    """One act in the corpus, with the identifiers needed to fetch and cite it."""

    key: str
    handle: str
    title: str
    short_title: str
    year: int
    collection: Collection = Collection.LEGAL_ACTS
    # Verified section count on the act page, recorded so drift is detectable.
    expected_sections: int | None = None
    # When set, only these section numbers are ingested (large codes).
    section_filter: tuple[str, ...] = field(default=())
    why: str = ""

    @property
    def citation_name(self) -> str:
        return f"{self.short_title}, {self.year}"


# --------------------------------------------------------------------------- verified acts
# Confirmed live: title and section count read back from India Code.
CONTRACT_ACT = ActSpec(
    key="contract_act_1872",
    handle="2187",
    title="Indian Contract Act, 1872",
    short_title="Indian Contract Act",
    year=1872,
    expected_sections=268,
    why="Core of the demo matter: breach of contract and compensation (ss. 73-75).",
)

BNS_2023 = ActSpec(
    key="bns_2023",
    handle="20062",
    title="Bharatiya Nyaya Sanhita, 2023",
    short_title="Bharatiya Nyaya Sanhita",
    year=2023,
    expected_sections=358,
    # 358 sections is too much for the MVP index; keep provisions a commercial dispute touches.
    section_filter=(
        "2", "3", "316", "318", "319", "320", "324", "326", "336", "338", "340",
        "351", "352", "356",
    ),
    why="Current criminal code. Cheating/criminal breach of trust often shadow contract disputes.",
)

BSA_2023 = ActSpec(
    key="bsa_2023",
    handle="20063",
    title="Bharatiya Sakshya Adhiniyam, 2023",
    short_title="Bharatiya Sakshya Adhiniyam",
    year=2023,
    expected_sections=170,
    section_filter=(
        "3", "4", "5", "6", "39", "56", "57", "58", "59", "60", "61", "62", "63",
        "64", "65", "94", "104", "105", "119", "141",
    ),
    why="Current evidence law. Governs proof of documents and electronic records - "
    "decisive when a matter turns on an agreement or notice.",
)

IT_ACT_2000 = ActSpec(
    key="it_act_2000",
    handle="1999",
    title="Information Technology Act, 2000",
    short_title="Information Technology Act",
    year=2000,
    expected_sections=125,
    section_filter=(
        "1", "2", "3", "4", "5", "10A", "11", "12", "13", "43", "43A", "65", "66",
        "66C", "66D", "72", "72A", "79", "85",
    ),
    why="Electronic contracts and records; s.10A validates contracts formed electronically.",
)

CPC_1908 = ActSpec(
    key="cpc_1908",
    handle="2191",
    title="Code of Civil Procedure, 1908",
    short_title="Code of Civil Procedure",
    year=1908,
    expected_sections=171,
    section_filter=(
        "9", "10", "11", "15", "16", "17", "18", "19", "20", "26", "34", "35", "35A",
        "80", "89", "96", "100", "151",
    ),
    why="Procedure for civil suits: jurisdiction, res judicata, interest, costs, s.89 ADR.",
)


CORPUS: tuple[ActSpec, ...] = (
    CONTRACT_ACT,
    BNS_2023,
    BSA_2023,
    IT_ACT_2000,
    CPC_1908,
)

_BY_KEY = {spec.key: spec for spec in CORPUS}


def get_act(key: str) -> ActSpec:
    try:
        return _BY_KEY[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown act '{key}'. Known: {', '.join(sorted(_BY_KEY))}"
        ) from exc


def all_acts() -> tuple[ActSpec, ...]:
    return CORPUS


def wants_section(spec: ActSpec, section_number: str) -> bool:
    """Whether a section is in scope for this act."""
    if not spec.section_filter:
        return True
    return section_number.strip().upper() in {s.upper() for s in spec.section_filter}


def estimated_provisions() -> int:
    """Rough size of the corpus, used to sanity-check against the brief's 100-300 target."""
    total = 0
    for spec in CORPUS:
        if spec.section_filter:
            total += len(spec.section_filter)
        elif spec.expected_sections:
            total += spec.expected_sections
    return total
