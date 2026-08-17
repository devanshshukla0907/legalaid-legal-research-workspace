"""Idempotent database seed.

Creates the Phase-1 personal workspace plus the RBAC scaffolding that firm mode will later
use. A personal workspace is modelled as an organization of one, so adding firm support is
authorisation logic and UI rather than a migration of core tables.

Also registers the knowledge sources so provenance rows exist before any ingestion runs, and
creates the demo matter used by the scripted demo.

Safe to run repeatedly: every step is a get-or-create.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from legalaid.core.logging import get_logger
from legalaid.db.base import create_all, session_scope
from legalaid.db.models import (
    KnowledgeSource,
    Matter,
    Membership,
    Organization,
    Permission,
    Role,
    RolePermission,
    User,
    Workspace,
)

logger = get_logger(__name__)

DEFAULT_ORG_SLUG = "personal"
DEFAULT_USER_EMAIL = "lawyer@legalaid.local"
DEMO_MATTER_REFERENCE = "LA-DEMO-001"

# --------------------------------------------------------------------------- RBAC vocabulary
# Defined now so Phase-1 rows are already permission-tagged; firm mode enforces them later.
PERMISSIONS: tuple[tuple[str, str], ...] = (
    ("matter.create", "Create a matter"),
    ("matter.read", "View a matter"),
    ("matter.update", "Edit matter metadata"),
    ("matter.delete", "Delete a matter"),
    ("document.upload", "Upload documents to a matter"),
    ("document.read", "Read documents and extracted text"),
    ("document.delete", "Delete a document"),
    ("research.run", "Run legal research queries"),
    ("draft.create", "Generate drafts"),
    ("draft.read", "Read drafts"),
    ("knowledge.ingest", "Ingest legislation and case law"),
    ("knowledge.read", "Browse the knowledge base"),
    ("trace.read", "Inspect pipeline traces"),
    ("admin.manage_members", "Manage organization members and roles"),
)

ALL_PERMISSION_CODES = tuple(code for code, _ in PERMISSIONS)

READ_ONLY_CODES = ("matter.read", "document.read", "draft.read", "knowledge.read", "trace.read")

# Roles a firm would need. The solo lawyer gets `owner`.
ROLES: dict[str, tuple[str, ...]] = {
    "owner": ALL_PERMISSION_CODES,
    "firm_admin": ALL_PERMISSION_CODES,
    "partner": tuple(c for c in ALL_PERMISSION_CODES if c != "admin.manage_members"),
    "associate": (
        "matter.create", "matter.read", "matter.update",
        "document.upload", "document.read",
        "research.run", "draft.create", "draft.read",
        "knowledge.read", "trace.read",
    ),
    "paralegal": (
        "matter.read", "document.upload", "document.read",
        "research.run", "draft.read", "knowledge.read",
    ),
    "researcher": (
        "matter.read", "document.read", "research.run",
        "knowledge.read", "knowledge.ingest", "trace.read",
    ),
    "read_only": READ_ONLY_CODES,
}


@dataclass(slots=True)
class SeedResult:
    organization_id: str
    user_id: str
    workspace_id: str
    demo_matter_id: str
    created: list[str]

    def summary(self) -> str:
        what = ", ".join(self.created) if self.created else "nothing new"
        return f"seed complete ({what})"


def _get_or_create_permissions(session: Session, created: list[str]) -> dict[str, Permission]:
    existing = {p.code: p for p in session.scalars(select(Permission)).all()}
    for code, description in PERMISSIONS:
        if code not in existing:
            permission = Permission(code=code, description=description)
            session.add(permission)
            existing[code] = permission
            created.append(f"permission:{code}")
    session.flush()
    return existing


def _get_or_create_roles(
    session: Session,
    organization: Organization,
    permissions: dict[str, Permission],
    created: list[str],
) -> dict[str, Role]:
    roles = {
        r.name: r
        for r in session.scalars(
            select(Role).where(Role.organization_id == organization.id)
        ).all()
    }
    for name, codes in ROLES.items():
        role = roles.get(name)
        if role is None:
            role = Role(organization_id=organization.id, name=name, is_system=True)
            session.add(role)
            session.flush()
            roles[name] = role
            created.append(f"role:{name}")

        granted = {
            rp.permission_id
            for rp in session.scalars(
                select(RolePermission).where(RolePermission.role_id == role.id)
            ).all()
        }
        for code in codes:
            permission = permissions[code]
            if permission.id not in granted:
                session.add(RolePermission(role_id=role.id, permission_id=permission.id))
    session.flush()
    return roles


def _knowledge_sources() -> tuple[dict[str, object], ...]:
    """Provenance records for every corpus we draw on.

    Licence and attribution are recorded here rather than in prose, because the UI cites them.
    """
    return (
        {
            "key": "india_code",
            "name": "India Code — Central Acts",
            "authority": "Government of India, Legislative Department",
            "source_url": "https://www.indiacode.nic.in/",
            "licence": "Government of India open content; reproduced with attribution",
            "attribution": (
                "Statutory text reproduced from India Code, the official legislation "
                "repository of the Government of India (Legislative Department)."
            ),
            "notes": (
                "Section text and amendment footnotes fetched per provision from the official "
                "site. Section-level provenance (act, section number, heading, source URL) is "
                "preserved for every chunk."
            ),
        },
        {
            "key": "supreme_court_judgments",
            "name": "Indian Supreme Court Judgments (open dataset)",
            "authority": "Supreme Court of India, via AWS Open Data",
            "source_url": "https://registry.opendata.aws/indian-supreme-court-judgments/",
            "licence": "CC-BY-4.0",
            "attribution": (
                "Judgment texts from the Indian Supreme Court Judgments open dataset "
                "(CC-BY-4.0), sourced from the official Supreme Court e-SCR portal."
            ),
            "notes": (
                "Chosen over IL-PCSR and ILDC/CJPE, which are CC-BY-NC and do not permit "
                "redistribution. Curated subset only."
            ),
        },
        {
            "key": "uploaded_documents",
            "name": "Lawyer-uploaded matter documents",
            "authority": "User-provided",
            "source_url": None,
            "licence": "Private to the workspace",
            "attribution": "Documents supplied by the lawyer for a specific matter.",
            "notes": "Never mixed with the public knowledge collections in the vector index.",
        },
    )


def _get_or_create_knowledge_sources(session: Session, created: list[str]) -> None:
    existing = {s.key for s in session.scalars(select(KnowledgeSource)).all()}
    for spec in _knowledge_sources():
        if spec["key"] in existing:
            continue
        session.add(KnowledgeSource(**spec))  # type: ignore[arg-type]
        created.append(f"source:{spec['key']}")
    session.flush()


def seed() -> SeedResult:
    """Create the personal workspace, RBAC scaffolding, sources and demo matter."""
    create_all()
    created: list[str] = []

    with session_scope() as session:
        organization = session.scalar(
            select(Organization).where(Organization.slug == DEFAULT_ORG_SLUG)
        )
        if organization is None:
            organization = Organization(
                name="Personal Workspace", slug=DEFAULT_ORG_SLUG, kind="personal"
            )
            session.add(organization)
            session.flush()
            created.append("organization")

        permissions = _get_or_create_permissions(session, created)
        roles = _get_or_create_roles(session, organization, permissions, created)

        user = session.scalar(select(User).where(User.email == DEFAULT_USER_EMAIL))
        if user is None:
            user = User(
                email=DEFAULT_USER_EMAIL,
                full_name="Demo Lawyer",
                # No credential is seeded: Phase 1 is single-user and unauthenticated by
                # design. Login belongs with firm mode, and a fake hash here would imply
                # an auth system that does not exist.
                password_hash=None,
            )
            session.add(user)
            session.flush()
            created.append("user")

        membership = session.scalar(
            select(Membership).where(
                Membership.organization_id == organization.id, Membership.user_id == user.id
            )
        )
        if membership is None:
            session.add(
                Membership(
                    organization_id=organization.id,
                    user_id=user.id,
                    role_id=roles["owner"].id,
                )
            )
            created.append("membership")

        workspace = session.scalar(
            select(Workspace).where(Workspace.organization_id == organization.id)
        )
        if workspace is None:
            workspace = Workspace(
                organization_id=organization.id,
                owner_user_id=user.id,
                name="My Practice",
                kind="personal",
            )
            session.add(workspace)
            session.flush()
            created.append("workspace")

        _get_or_create_knowledge_sources(session, created)

        matter = session.scalar(
            select(Matter).where(Matter.reference_no == DEMO_MATTER_REFERENCE)
        )
        if matter is None:
            matter = Matter(
                workspace_id=workspace.id,
                title="Contract Dispute — Breach and Compensation",
                reference_no=DEMO_MATTER_REFERENCE,
                matter_type="Civil — contract",
                court="District Court (proposed)",
                parties={"claimant": "To be confirmed", "respondent": "To be confirmed"},
                description=(
                    "Demo matter for the scripted walkthrough. A supplier alleges breach of a "
                    "supply agreement and claims compensation for loss; the other side disputes "
                    "quantum and relies on a penalty clause. Facts and parties are filled in "
                    "from documents the lawyer uploads — nothing is pre-fabricated here."
                ),
                status="active",
                is_demo=True,
            )
            session.add(matter)
            session.flush()
            created.append("demo_matter")

        result = SeedResult(
            organization_id=organization.id,
            user_id=user.id,
            workspace_id=workspace.id,
            demo_matter_id=matter.id,
            created=created,
        )

    logger.info(
        "seed_complete",
        organization=result.organization_id,
        workspace=result.workspace_id,
        created=len(result.created),
    )
    return result


def default_workspace_id() -> str:
    """The workspace the Phase-1 single-user UI operates in."""
    with session_scope() as session:
        workspace = session.scalar(select(Workspace).order_by(Workspace.created_at))
        if workspace is None:
            raise RuntimeError("No workspace found. Run the seed first: python -m legalaid.db.seed")
        return workspace.id


if __name__ == "__main__":
    outcome = seed()
    print(outcome.summary())
    print(f"  organization : {outcome.organization_id}")
    print(f"  user         : {outcome.user_id}")
    print(f"  workspace    : {outcome.workspace_id}")
    print(f"  demo matter  : {outcome.demo_matter_id}")
