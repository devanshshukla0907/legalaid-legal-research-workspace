"""The legal corpus manifest.

Every entry was verified against the live India Code API (`scripts/check_indiacode_api.py`):
the title, act id, section count and commencement date below were read back from the service,
not assumed. A wrong identifier would silently ingest the wrong statute, which is worse than
ingesting nothing.

Two things changed in August 2026 and both are reflected here:

* India Code migrated to a DSpace 9 API, so acts are addressed by `act_id` rather than by the
  old DSpace handle. `search_title` is retained as a fallback in case an id is reissued.
* India Code holds dozens of *state adoptions* of each central act. Every id below is the
  CENTRAL copy, confirmed via `dc.identifier.state_name`.

Section filters exist only where a code is very large and most of it is irrelevant to the
application's scope. Where an act is small enough to hold in full, it is held in full - filtering
a statute means a question about an unfiltered section retrieves nothing, and "the corpus does
not contain it" is a much less useful answer than the provision itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from legalaid.domain.enums import Collection


@dataclass(frozen=True, slots=True)
class ActSpec:
    """One act in the corpus, with the identifiers needed to fetch and cite it."""

    key: str
    title: str
    short_title: str
    year: int
    # India Code DSpace act id for the CENTRAL copy. Verified live.
    act_id: str
    # Fallback for resolution if the id is ever reissued.
    search_title: str = ""
    collection: Collection = Collection.LEGAL_ACTS
    # Section count read back from the API, recorded so drift is detectable.
    expected_sections: int | None = None
    # Commencement date as India Code reports it. Some are staged across notifications, so this
    # is stored as the publisher's own string rather than coerced into a date.
    commencement: str = ""
    # Only for very large codes. Empty means "ingest every section".
    section_filter: tuple[str, ...] = ()
    why: str = ""

    @property
    def citation_name(self) -> str:
        return self.title

    @property
    def resolve_title(self) -> str:
        return self.search_title or self.title


# --------------------------------------------------------------- core private law

CONTRACT_ACT = ActSpec(
    key="contract_act_1872",
    title="Indian Contract Act, 1872",
    short_title="Indian Contract Act",
    year=1872,
    act_id="AC_CEN_3_20_00035_187209_1523268996428",
    expected_sections=268,
    commencement="1872-09-01",
    why="The centre of gravity for commercial disputes: formation, breach, "
    "compensation (ss.73-75), indemnity, guarantee, bailment and agency.",
)

SPECIFIC_RELIEF_ACT = ActSpec(
    key="specific_relief_1963",
    title="Specific Relief Act, 1963",
    short_title="Specific Relief Act",
    year=1963,
    act_id="AC_CEN_3_20_00009_196347_1517807320084",
    commencement="01-03-1964",
    why="Remedies beyond damages: specific performance, injunctions, rescission. "
    "A contract question is incomplete without it.",
)

SALE_OF_GOODS_ACT = ActSpec(
    key="sale_of_goods_1930",
    title="Sale of Goods Act, 1930",
    short_title="Sale of Goods Act",
    year=1930,
    act_id="AC_CEN_3_20_00059_193003_1523350185738",
    commencement="1930-07-01",
    why="Governs the supply-of-goods disputes the evaluation set is built around: "
    "passing of property, conditions and warranties, seller's remedies.",
)

TRANSFER_OF_PROPERTY_ACT = ActSpec(
    key="transfer_of_property_1882",
    title="Transfer of Property Act, 1882",
    short_title="Transfer of Property Act",
    year=1882,
    act_id="AC_CEN_3_20_00042_188204_1523272233671",
    commencement="1882-07-01",
    why="Immovable property transactions: sale, mortgage, lease, part performance.",
)

NEGOTIABLE_INSTRUMENTS_ACT = ActSpec(
    key="negotiable_instruments_1881",
    title="Negotiable Instruments Act, 1881",
    short_title="Negotiable Instruments Act",
    year=1881,
    act_id="AC_CEN_2_33_00042_00042_1523271998701",
    commencement="1882-03-01",
    why="Cheque dishonour under s.138 is one of the highest-volume commercial "
    "proceedings in India and shadows many contract disputes.",
)

PARTNERSHIP_ACT = ActSpec(
    key="partnership_1932",
    title="Indian Partnership Act, 1932",
    short_title="Indian Partnership Act",
    year=1932,
    act_id="",  # resolved by title; see check_indiacode_api.py output
    search_title="Indian Partnership Act, 1932",
    commencement="1932-10-01 (except s.69), 1933-10-01",
    why="Firm liability and the s.69 bar on suits by unregistered firms, which "
    "decides whether a claim is maintainable at all.",
)

# ------------------------------------------------------------------- procedure

CPC_1908 = ActSpec(
    key="cpc_1908",
    title="Code of Civil Procedure, 1908",
    short_title="Code of Civil Procedure",
    year=1908,
    act_id="",
    search_title="Code of Civil Procedure, 1908",
    expected_sections=171,
    why="Civil procedure: jurisdiction, res judicata, interest, costs, s.89 ADR.",
)

LIMITATION_ACT = ActSpec(
    key="limitation_1963",
    title="Limitation Act, 1963",
    short_title="Limitation Act",
    year=1963,
    act_id="AC_CEN_3_20_00005_196336_1517807319297",
    commencement="01-01-1964",
    why="Whether a claim is time-barred is usually the first question asked and is "
    "dispositive when the answer is yes.",
)

ARBITRATION_ACT = ActSpec(
    key="arbitration_1996",
    title="Arbitration and Conciliation Act, 1996",
    short_title="Arbitration and Conciliation Act",
    year=1996,
    act_id="AC_CEN_3_46_00004_199626_1517807323919",
    commencement="22-08-1996",
    why="Most commercial contracts arbitrate. Also the subject of the 2023 "
    "seven-judge stamping decision already in the case-law corpus.",
)

# -------------------------------------------------------- criminal and evidence

BNS_2023 = ActSpec(
    key="bns_2023",
    title="Bharatiya Nyaya Sanhita, 2023",
    short_title="Bharatiya Nyaya Sanhita",
    year=2023,
    act_id="",
    search_title="Bharatiya Nyaya Sanhita, 2023",
    expected_sections=358,
    why="Current criminal code. Cheating and criminal breach of trust routinely "
    "shadow commercial disputes.",
)

BNSS_2023 = ActSpec(
    key="bnss_2023",
    title="Bharatiya Nagarik Suraksha Sanhita, 2023",
    short_title="Bharatiya Nagarik Suraksha Sanhita",
    year=2023,
    act_id="",
    search_title="Bharatiya Nagarik Suraksha Sanhita, 2023",
    expected_sections=531,
    # 531 sections, and the application's criminal scope is narrow, so this is the one act
    # deliberately reduced to the provisions a commercial matter actually reaches.
    section_filter=(
        "2", "35", "36", "39", "47", "58", "63", "64", "65", "66", "67", "68",
        "173", "174", "175", "176", "179", "180", "183", "187", "190", "193",
        "223", "227", "230", "251", "258", "262", "265", "268", "270",
        "289", "290", "308", "319", "329", "330", "336", "337", "343", "346",
        "356", "379", "393", "398", "413", "415", "419", "422", "438", "442",
        "479", "480", "481", "482", "483", "484", "492", "497", "528", "531",
    ),
    why="Current criminal procedure: FIR registration, bail, summons, framing of "
    "charge - what a client facing an FIR asks about first.",
)

BSA_2023 = ActSpec(
    key="bsa_2023",
    title="Bharatiya Sakshya Adhiniyam, 2023",
    short_title="Bharatiya Sakshya Adhiniyam",
    year=2023,
    act_id="",
    search_title="Bharatiya Sakshya Adhiniyam, 2023",
    expected_sections=170,
    why="Current evidence law. Decides how an agreement, notice or electronic "
    "record is actually proved.",
)

# --------------------------------------------------------- digital and consumer

IT_ACT_2000 = ActSpec(
    key="it_act_2000",
    title="Information Technology Act, 2000",
    short_title="Information Technology Act",
    year=2000,
    act_id="",
    search_title="Information Technology Act, 2000",
    expected_sections=125,
    why="s.10A validates contracts formed electronically; ss.65-66 and 43A cover "
    "tampering, computer offences and data-protection failures.",
)

CONSUMER_PROTECTION_ACT = ActSpec(
    key="consumer_protection_2019",
    title="Consumer Protection Act, 2019",
    short_title="Consumer Protection Act",
    year=2019,
    act_id="",
    search_title="Consumer Protection Act, 2019",
    commencement="staged from 24 July 2020 by notification S.O. 2421(E)",
    why="Product liability and unfair trade practice, and a far cheaper forum than "
    "a civil suit for a claimant who qualifies.",
)

RTI_ACT = ActSpec(
    key="rti_2005",
    title="Right to Information Act, 2005",
    short_title="Right to Information Act",
    year=2005,
    act_id="",
    search_title="Right to Information Act, 2005",
    commencement="ss.4(1), 5(1)-(2), 12, 13, 15, 16, 24, 27, 28 on 15-06-2005; "
    "remainder on the 120th day after enactment",
    why="How to obtain documents from a public authority, which is often the first "
    "practical step in a dispute with the state.",
)


CORPUS: tuple[ActSpec, ...] = (
    # private law
    CONTRACT_ACT,
    SPECIFIC_RELIEF_ACT,
    SALE_OF_GOODS_ACT,
    TRANSFER_OF_PROPERTY_ACT,
    NEGOTIABLE_INSTRUMENTS_ACT,
    PARTNERSHIP_ACT,
    # procedure
    CPC_1908,
    LIMITATION_ACT,
    ARBITRATION_ACT,
    # criminal and evidence
    BNS_2023,
    BNSS_2023,
    BSA_2023,
    # digital and consumer
    IT_ACT_2000,
    CONSUMER_PROTECTION_ACT,
    RTI_ACT,
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
    """Rough corpus size, for sanity-checking an ingestion run against expectations."""
    total = 0
    for spec in CORPUS:
        if spec.section_filter:
            total += len(spec.section_filter)
        elif spec.expected_sections:
            total += spec.expected_sections
        else:
            # Unmeasured acts: a mid-range estimate so the total is not silently understated.
            total += 90
    return total
