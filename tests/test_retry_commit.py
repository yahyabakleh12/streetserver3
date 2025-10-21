import os

import pytest
from sqlalchemy.exc import OperationalError

# Use a dedicated SQLite database for these tests
TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from db import Base, engine, SessionLocal  # noqa: E402
from models import User  # noqa: E402
from main import _retry_commit  # noqa: E402


@pytest.fixture(autouse=True)
def setup_database():
    """Ensure a clean database for each test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    try:
        os.remove("test.db")
    except FileNotFoundError:
        pass


def test_retry_commit_new_entity(monkeypatch):
    session = SessionLocal()
    user = User(username="u1", hashed_password="pw")
    session.add(user)

    def fail_commit():
        raise OperationalError("", {}, Exception("fail"))

    monkeypatch.setattr(session, "commit", fail_commit)

    _retry_commit(user, session)

    session2 = SessionLocal()
    try:
        db_user = session2.query(User).filter_by(username="u1").one()
    finally:
        session2.close()
    assert db_user.username == "u1"


def test_retry_commit_existing_entity(monkeypatch):
    # Persist a user first
    session = SessionLocal()
    user = User(username="u1", hashed_password="pw")
    session.add(user)
    session.commit()
    session.close()

    session = SessionLocal()
    user = session.query(User).filter_by(username="u1").one()
    user.hashed_password = "newpw"

    def fail_commit():
        raise OperationalError("", {}, Exception("fail"))

    monkeypatch.setattr(session, "commit", fail_commit)

    _retry_commit(user, session)

    session2 = SessionLocal()
    try:
        users = session2.query(User).all()
        assert len(users) == 1
        assert users[0].hashed_password == "newpw"
    finally:
        session2.close()
