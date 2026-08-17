"""List the gateway's HTTP routes."""

from __future__ import annotations

from legalaid.services.gateway.app import app

routes = [r for r in app.routes if hasattr(r, "methods")]
print(f"{len(routes)} routes")
for route in sorted(routes, key=lambda r: r.path):
    methods = ",".join(sorted(m for m in route.methods if m != "HEAD"))
    print(f"  {methods:8} {route.path}")
