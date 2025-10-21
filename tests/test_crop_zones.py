import os
from fastapi.testclient import TestClient
import pytest

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from db import Base, engine, SessionLocal
from main import app, get_password_hash
from models import Location, Zone, Pole, Camera, CropZone, User

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


def test_create_and_list_crop_zones(client, sample_camera):
    resp = client.post(
        "/crop-zones",
        json={"camera_id": sample_camera, "points": [{"x":0,"y":0},{"x":10,"y":0},{"x":10,"y":10}]},
    )
    assert resp.status_code == 200
    zone_id = resp.json()["id"]

    session = SessionLocal()
    zone_row = session.query(CropZone).get(zone_id)
    assert zone_row is not None
    assert zone_row.camera_id == sample_camera
    assert isinstance(zone_row.points, list)
    session.close()

    resp = client.get(f"/cameras/{sample_camera}/crop-zones")
    assert resp.status_code == 200
    zones = resp.json()
    assert len(zones) == 1
    assert zones[0]["id"] == zone_id

    resp = client.get("/crop-zones")
    assert resp.status_code == 200
    assert any(z["id"] == zone_id for z in resp.json())

