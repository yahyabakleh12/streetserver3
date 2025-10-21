import os
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from db import Base, engine, SessionLocal
from main import app, get_password_hash
from models import Location, Zone, Pole, Camera, ClipRequest, User

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

session = SessionLocal()
user = User(username="test", hashed_password=get_password_hash("secret"))
session.add(user)
session.commit()
session.close()

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        resp = c.post(
            "/token",
            data={"username": "test", "password": "secret"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token = resp.json()["access_token"]
        c.headers.update({"Authorization": f"Bearer {token}"})
        yield c
    Base.metadata.drop_all(bind=engine)
    try:
        os.remove("test.db")
    except FileNotFoundError:
        pass

@pytest.fixture()
def sample_camera(tmp_path):
    session = SessionLocal()
    loc = Location(name="Loc", code="L1", portal_name="u", portal_password="p", ip_schema="ip", camera_user="user", camera_pass="pass")
    session.add(loc)
    session.commit()
    zone = Zone(code="Z1", location_id=loc.id)
    session.add(zone)
    session.commit()
    pole = Pole(zone_id=zone.id, code="P1", location_id=loc.id)
    session.add(pole)
    session.commit()
    cam = Camera(pole_id=pole.id, api_code="C1", p_ip="127.0.0.1", portal_id=1)
    session.add(cam)
    session.commit()
    session.close()
    return cam.id


def test_clip_request_flow(client, sample_camera, tmp_path):
    clip_file = tmp_path / "clip.mp4"
    clip_file.write_bytes(b"data")
    start = datetime(2025, 1, 1, 0, 0, 0)
    end = datetime(2025, 1, 1, 0, 0, 10)
    with patch("main.request_camera_clip", return_value=str(clip_file)), \
         patch("main.is_valid_mp4", return_value=True):
        resp = client.post(
            "/clip-requests",
            json={"camera_id": sample_camera, "start": start.isoformat(), "end": end.isoformat()},
        )
    assert resp.status_code == 200
    data = resp.json()
    req_id = data["id"]
    assert data["status"] == "PENDING"

    session = SessionLocal()
    req = session.query(ClipRequest).get(req_id)
    assert req.status == "COMPLETED"
    assert req.clip_path == str(clip_file)
    session.close()

    resp = client.get("/clip-requests")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    resp = client.delete(f"/clip-requests/{req_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    assert not os.path.exists(clip_file)
    session = SessionLocal()
    assert session.query(ClipRequest).get(req_id) is None
    session.close()

