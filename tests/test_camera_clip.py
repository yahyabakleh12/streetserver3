import os
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from db import Base, engine, SessionLocal
from main import app, get_password_hash
from models import Location, Zone, Pole, Camera, User

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
    loc = Location(
        name="Loc",
        code="L1",
        portal_name="u",
        portal_password="p",
        ip_schema="ip",
        camera_user="user",
        camera_pass="pass",
    )
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


def test_get_camera_clip_success(client, sample_camera, tmp_path):
    clip_file = tmp_path / "clip.mp4"
    clip_file.write_bytes(b"data")
    start = "2025-01-01T00:00:00"
    end = "2025-01-01T00:00:10"
    with patch("main.request_camera_clip", return_value=str(clip_file)), \
         patch("main.is_valid_mp4", return_value=True):
        resp = client.get(f"/cameras/{sample_camera}/clip?start={start}&end={end}")
    assert resp.status_code == 200
    assert resp.content == b"data"


def test_get_camera_clip_not_found(client):
    start = "2025-01-01T00:00:00"
    end = "2025-01-01T00:00:10"
    resp = client.get(f"/cameras/999/clip?start={start}&end={end}")
    assert resp.status_code == 404


def test_get_camera_clip_bad_time(client, sample_camera):
    resp = client.get(f"/cameras/{sample_camera}/clip?start=bad&end=2025-01-01T00:00:10")
    assert resp.status_code == 400


def test_get_camera_clip_invalid_video(client, sample_camera, tmp_path):
    clip_file = tmp_path / "clip.mp4"
    clip_file.write_bytes(b"bad")
    start = "2025-01-01T00:00:00"
    end = "2025-01-01T00:00:10"
    with patch("main.request_camera_clip", return_value=str(clip_file)), \
         patch("main.is_valid_mp4", return_value=False):
        resp = client.get(f"/cameras/{sample_camera}/clip?start={start}&end={end}")
    assert resp.status_code == 500


def test_get_camera_clip_fetch_fail(client, sample_camera):
    start = "2025-01-01T00:00:00"
    end = "2025-01-01T00:00:10"
    with patch("main.request_camera_clip", return_value=None):
        resp = client.get(f"/cameras/{sample_camera}/clip?start={start}&end={end}")
    assert resp.status_code == 500
