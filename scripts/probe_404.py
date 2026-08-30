"""Why does the stored source_url 404 now when it worked during ingestion?"""

from __future__ import annotations

import urllib.error
import urllib.request

from legalaid.ingestion.india_code import USER_AGENT

URL = (
    "https://www.indiacode.nic.in/SectionPageContent"
    "?actid=AC_CEN_3_20_00035_187209_1523268996428&sectionID=38677"
)

BROWSER = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

VARIANTS = [
    ("module UA only", {"User-Agent": USER_AGENT}),
    ("browser UA", {"User-Agent": BROWSER}),
    (
        "module UA + Referer",
        {
            "User-Agent": USER_AGENT,
            "Referer": "https://www.indiacode.nic.in/handle/123456789/2187",
        },
    ),
    (
        "module UA + XHR headers",
        {
            "User-Agent": USER_AGENT,
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://www.indiacode.nic.in/handle/123456789/2187",
        },
    ),
]

for label, headers in VARIANTS:
    request = urllib.request.Request(URL, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = response.read()
        print(f"  {label:<26} HTTP {response.status}  {len(body):,} bytes")
    except urllib.error.HTTPError as exc:
        print(f"  {label:<26} HTTP {exc.code}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:<26} ERR {type(exc).__name__}")

# Is the act page itself still reachable? That tells us whether the whole site changed
# or just this endpoint's access rules.
print()
for label, url in [
    ("act page", "https://www.indiacode.nic.in/handle/123456789/2187"),
    ("site root", "https://www.indiacode.nic.in/"),
]:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=45) as response:
            print(f"  {label:<26} HTTP {response.status}  {len(response.read()):,} bytes")
    except urllib.error.HTTPError as exc:
        print(f"  {label:<26} HTTP {exc.code}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:<26} ERR {type(exc).__name__}")
