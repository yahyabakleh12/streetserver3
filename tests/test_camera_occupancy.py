import os
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from db import Base, engine, SessionLocal
from main import app, get_password_hash
from models import Location, Zone, Pole, Camera, Spot, Ticket, User

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
def sample_data():
    session = SessionLocal()
    loc = Location(name="Loc", code="L1", portal_name="u", portal_password="p", ip_schema="ip")
    loc2 = Location(name="Other", code="L2", portal_name="u", portal_password="p", ip_schema="ip")
    session.add_all([loc, loc2])
    session.commit()

    zone = Zone(code="Z1", location_id=loc.id)
    zone2 = Zone(code="Z2", location_id=loc2.id)
    session.add_all([zone, zone2])
    session.commit()

    pole = Pole(zone_id=zone.id, code="P1", location_id=loc.id)
    pole2 = Pole(zone_id=zone2.id, code="P2", location_id=loc2.id)
    session.add_all([pole, pole2])
    session.commit()

    cam1 = Camera(pole_id=pole.id, api_code="C1", p_ip="1", portal_id=1)
    cam2 = Camera(pole_id=pole.id, api_code="C2", p_ip="2", portal_id=2)
    cam3 = Camera(pole_id=pole2.id, api_code="C3", p_ip="3", portal_id=3)
    session.add_all([cam1, cam2, cam3])
    session.commit()

    s1 = Spot(camera_id=cam1.id, spot_number=1, p1_x=0, p1_y=0,
              p2_x=1, p2_y=0, p3_x=1, p3_y=1, p4_x=0, p4_y=1, status=1)
    s2 = Spot(camera_id=cam1.id, spot_number=2, p1_x=0, p1_y=0,
              p2_x=1, p2_y=0, p3_x=1, p3_y=1, p4_x=0, p4_y=1, status=1)
    s3 = Spot(camera_id=cam2.id, spot_number=1, p1_x=0, p1_y=0,
              p2_x=1, p2_y=0, p3_x=1, p3_y=1, p4_x=0, p4_y=1, status=1)
    s4 = Spot(camera_id=cam2.id, spot_number=2, p1_x=0, p1_y=0,
              p2_x=1, p2_y=0, p3_x=1, p3_y=1, p4_x=0, p4_y=1, status=0)
    session.add_all([s1, s2, s3, s4])
    session.commit()

    t1 = Ticket(camera_id=cam1.id, spot_number=1, plate_number="A", entry_time=datetime.utcnow())
    t2 = Ticket(camera_id=cam1.id, spot_number=2, plate_number="B", entry_time=datetime.utcnow())
    t3 = Ticket(camera_id=cam2.id, spot_number=1, plate_number="C", entry_time=datetime.utcnow())
    t4 = Ticket(camera_id=cam2.id, spot_number=2, plate_number="D", entry_time=datetime.utcnow(), exit_time=datetime.utcnow())
    session.add_all([t1, t2, t3, t4])
    session.commit()
    cam1_id = cam1.id
    cam2_id = cam2.id
    cam3_id = cam3.id
    loc1_id = loc.id
    loc2_id = loc2.id
    session.close()
    return cam1_id, cam2_id, cam3_id, loc1_id, loc2_id


def test_camera_occupancy(client, sample_data):
    cam1_id, cam2_id, _, _, _ = sample_data
    resp = client.get("/camera-occupancy")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2

    info = {d["id"]: d for d in data}
    assert info[cam1_id]["spot_count"] == 2
    assert info[cam1_id]["occupied_count"] == 2
    assert sorted(info[cam1_id]["occupied_spots"]) == [1, 2]
    assert info[cam1_id]["spots"] == {"1": 1, "2": 1}

    assert info[cam2_id]["spot_count"] == 2
    assert info[cam2_id]["occupied_count"] == 1
    assert info[cam2_id]["occupied_spots"] == [1]
    assert info[cam2_id]["spots"] == {"1": 1, "2": 0}


def test_camera_occupancy_location_filter(client, sample_data):
    cam1_id, cam2_id, cam3_id, loc1_id, loc2_id = sample_data
    resp = client.get(f"/camera-occupancy/{loc1_id}")
    assert resp.status_code == 200
    data = resp.json()
    ids = {c["id"] for c in data}
    assert cam1_id in ids and cam2_id in ids
    assert cam3_id not in ids

    info = {d["id"]: d for d in data}
    assert info[cam1_id]["spot_count"] == 2
    assert info[cam2_id]["spot_count"] == 2
