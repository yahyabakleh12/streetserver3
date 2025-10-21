import os
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
def sample_camera():
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


def test_get_camera_frame_success(client, sample_camera):
    with patch("main.fetch_camera_frame", return_value=b"img"):
        resp = client.get(f"/cameras/{sample_camera}/frame")
    assert resp.status_code == 200
    assert resp.content == b"img"
    assert resp.headers["content-type"] == "image/jpeg"


def test_get_camera_frame_not_found(client):
    resp = client.get("/cameras/999/frame")
    assert resp.status_code == 404


def test_get_camera_frame_error(client, sample_camera):
    with patch("main.fetch_camera_frame", side_effect=Exception("boom")):
        resp = client.get(f"/cameras/{sample_camera}/frame")
    assert resp.status_code == 500
