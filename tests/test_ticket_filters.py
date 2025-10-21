import os
from datetime import datetime
import pytest
from fastapi.testclient import TestClient

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from db import Base, engine, SessionLocal
from main import app, get_password_hash
from models import Location, Zone, Pole, Camera, Ticket, User

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

session = SessionLocal()
user = User(username="test", hashed_password=get_password_hash("secret"))
session.add(user)
session.commit()

loc = Location(name="Loc", code="L1", portal_name="u", portal_password="p", ip_schema="ip")
session.add(loc)
session.commit()
zone = Zone(code="Z1", location_id=loc.id)
session.add(zone)
session.commit()
pole = Pole(zone_id=zone.id, code="P1", location_id=loc.id)
session.add(pole)
session.commit()
cam = Camera(pole_id=pole.id, api_code="C1", p_ip="1", portal_id=1)
session.add(cam)
session.commit()

session.add_all([
    Ticket(
        camera_id=cam.id,
        spot_number=1,
        plate_number="ABC123",
        plate_code="1",
        plate_city="DXB",
        entry_time=datetime(2024, 1, 1, 12, 0),
    ),
    Ticket(
        camera_id=cam.id,
        spot_number=2,
        plate_number="DEF456",
        plate_code="2",
        plate_city="AUH",
        entry_time=datetime(2024, 1, 2, 12, 0),
    ),
])
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

def test_filter_by_plate_and_time(client):
    resp = client.get(
        "/tickets",
        params={
            "plate_number": "ABC123",
            "entry_start": "2024-01-01T00:00:00",
            "entry_end": "2024-01-01T23:59:59",
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    assert data[0]["plate_number"] == "ABC123"

def test_filter_by_spot_and_code(client):
    resp = client.get(
        "/tickets",
        params={"spot_number": 2, "plate_code": "2"},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    assert data[0]["spot_number"] == 2
    assert data[0]["plate_code"] == "2"
