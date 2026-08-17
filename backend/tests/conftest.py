"""Test fixtures.

Every test runs against a temporary database and index directory so tests never touch the
developer's real data.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_environment() -> Iterator[None]:
    tmp = Path(tempfile.mkdtemp(prefix="legalaid_tests_"))
    os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{(tmp / 'test.db').as_posix()}"
    os.environ["DATA_DIR"] = str(tmp / "data")
    os.environ["UPLOAD_DIR"] = str(tmp / "uploads")
    os.environ["INDEX_DIR"] = str(tmp / "indexes")
    os.environ["KNOWLEDGE_DIR"] = str(tmp / "knowledge")
    os.environ["ENVIRONMENT"] = "test"
    os.environ["SERVICE_TRANSPORT"] = "inprocess"
    os.environ["ACTIVE_PHASE"] = "5"

    from legalaid.config import get_settings
    from legalaid.db.base import reset_engine

    get_settings.cache_clear()
    reset_engine()
    yield


@pytest.fixture()
def db() -> Iterator[None]:
    from legalaid.db.base import Base, create_all, get_engine

    create_all()
    yield
    Base.metadata.drop_all(bind=get_engine())
