"""Why did three acts not resolve? Search more loosely and show what India Code actually holds."""

from __future__ import annotations

from legalaid.ingestion import india_code_api as ic

QUERIES = [
    "Consumer Protection Act",
    "Right to Information",
    "Partnership Act",
    "Bharatiya Nagarik Suraksha Sanhita",
    "Insolvency and Bankruptcy Code",
    "Companies Act",
]


def main() -> None:
    for query in QUERIES:
        print(f"\n=== {query} ===")
        try:
            # central_only off, so state copies are visible and the reason for a miss is clear.
            refs = ic.find_acts(query, central_only=False, limit=8)
        except Exception as exc:  # noqa: BLE001
            print(f"  error: {type(exc).__name__}: {exc}")
            continue
        if not refs:
            print("  nothing returned")
            continue
        for ref in refs:
            marker = "CENTRAL" if ref.is_central else ref.state_name
            print(
                f"  [{marker:<14}] {ref.title[:52]:<54} year={ref.year} "
                f"repealed={ref.repealed}"
            )


if __name__ == "__main__":
    main()
