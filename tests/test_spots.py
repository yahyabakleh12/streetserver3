import os
from fastapi.testclient import TestClient
import pytest

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from db import Base, engine, SessionLocal
from main import app, get_password_hash
from models import Location, Zone, Pole, Camera, Spot, User

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
    loc = Location(name="Loc", code="L1", portal_name="u", portal_password="p", ip_schema="ip")
    session.add(loc)
    session.commit()
    zone = Zone(code="Z1", location_id=loc.id)
    session.add(zone)
    session.commit()
    pole = Pole(zone_id=zone.id, code="P1", location_id=loc.id)
    session.add(pole)
    session.commit()
    cam = Camera(pole_id=pole.id, api_code="C1", p_ip="ip", portal_id=1)
    session.add(cam)
    session.commit()
    cam_id = cam.id
    session.close()
    return cam_id


def test_create_and_list_spots(client, sample_camera):
    resp = client.post(
        "/spots",
        json={
            "camera_id": sample_camera,
            "spot_number": 1,
            "p1_x": 0,
            "p1_y": 0,
            "p2_x": 10,
            "p2_y": 0,
            "p3_x": 10,
            "p3_y": 10,
            "p4_x": 0,
            "p4_y": 10,
        },
    )
    assert resp.status_code == 200
    spot_id = resp.json()["id"]

    session = SessionLocal()
    spot_row = session.query(Spot).get(spot_id)
    assert spot_row is not None
    assert spot_row.camera_id == sample_camera
    assert spot_row.spot_number == 1
    assert (spot_row.p1_x, spot_row.p1_y) == (0, 0)
    assert (spot_row.p2_x, spot_row.p2_y) == (10, 0)
    assert (spot_row.p3_x, spot_row.p3_y) == (10, 10)
    assert (spot_row.p4_x, spot_row.p4_y) == (0, 10)
    assert spot_row.status == 0
    session.close()

    resp = client.get(f"/cameras/{sample_camera}/spots")
    assert resp.status_code == 200
    spots = resp.json()
    assert len(spots) == 1
    assert spots[0]["id"] == spot_id

    resp = client.get("/spots")
    assert resp.status_code == 200
    assert any(s["id"] == spot_id for s in resp.json())


def test_create_spot_camera_not_found(client):
    resp = client.post(
        "/spots",
        json={
            "camera_id": 999,
            "spot_number": 1,
            "p1_x": 0,
            "p1_y": 0,
            "p2_x": 1,
            "p2_y": 0,
            "p3_x": 1,
            "p3_y": 1,
            "p4_x": 0,
            "p4_y": 1,
        },
    )
    assert resp.status_code == 404


def test_get_spot_and_not_found(client, sample_camera):
    resp = client.post(
        "/spots",
        json={
            "camera_id": sample_camera,
            "spot_number": 2,
            "p1_x": 5,
            "p1_y": 5,
            "p2_x": 15,
            "p2_y": 5,
            "p3_x": 15,
            "p3_y": 15,
            "p4_x": 5,
            "p4_y": 15,
        },
    )
    assert resp.status_code == 200
    spot_id = resp.json()["id"]

    resp = client.get(f"/spots/{spot_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == spot_id
    assert data["camera_id"] == sample_camera
    assert data["spot_number"] == 2

    resp = client.get("/spots/999")
    assert resp.status_code == 404

