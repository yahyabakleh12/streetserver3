import os
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from db import Base, engine, SessionLocal
from main import app, get_password_hash
from models import Location, Zone, Pole, Camera, Ticket, ManualReview, User

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

session = SessionLocal()
user = User(username="test", hashed_password=get_password_hash("secret"))
session.add(user)
session.commit()

# sample data
loc = Location(name="Loc", code="L1", portal_name="u", portal_password="p", ip_schema="ip")
session.add(loc)
session.commit()

zone1 = Zone(code="Z1", location_id=loc.id)
zone2 = Zone(code="Z2", location_id=loc.id)
session.add_all([zone1, zone2])
session.commit()

pole1 = Pole(zone_id=zone1.id, code="P1", location_id=loc.id)
pole2 = Pole(zone_id=zone2.id, code="P2", location_id=loc.id)
session.add_all([pole1, pole2])
session.commit()

cam1 = Camera(pole_id=pole1.id, api_code="C1", p_ip="1", portal_id=1)
cam2 = Camera(pole_id=pole1.id, api_code="C2", p_ip="2", portal_id=2)
cam3 = Camera(pole_id=pole2.id, api_code="C3", p_ip="3", portal_id=3)
session.add_all([cam1, cam2, cam3])
session.commit()

# tickets
session.add_all([
    Ticket(camera_id=cam1.id, spot_number=1, plate_number="A", entry_time=datetime.utcnow()),
    Ticket(camera_id=cam1.id, spot_number=2, plate_number="B", entry_time=datetime.utcnow()),
    Ticket(camera_id=cam2.id, spot_number=1, plate_number="C", entry_time=datetime.utcnow()),
    Ticket(camera_id=cam3.id, spot_number=1, plate_number="D", entry_time=datetime.utcnow()),
    Ticket(camera_id=cam3.id, spot_number=2, plate_number="E", entry_time=datetime.utcnow()),
    Ticket(camera_id=cam3.id, spot_number=3, plate_number="F", entry_time=datetime.utcnow()),
])

# manual reviews
session.add_all([
    ManualReview(
        camera_id=cam1.id,
        spot_number=1,
        event_time=datetime.utcnow(),
        image_path="a.jpg",
        clip_path=None,
        ticket_id=None,
        plate_status="UNREAD",
        plate_image="p.jpg",
        snapshot_folder="f",
        review_status="PENDING",
    ),
    ManualReview(
        camera_id=cam3.id,
        spot_number=1,
        event_time=datetime.utcnow(),
        image_path="b.jpg",
        clip_path=None,
        ticket_id=None,
        plate_status="UNREAD",
        plate_image="p.jpg",
        snapshot_folder="f",
        review_status="PENDING",
    ),
    ManualReview(
        camera_id=cam3.id,
        spot_number=2,
        event_time=datetime.utcnow(),
        image_path="c.jpg",
        clip_path=None,
        ticket_id=None,
        plate_status="UNREAD",
        plate_image="p.jpg",
        snapshot_folder="f",
        review_status="PENDING",
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


def test_location_stats(client):
    response = client.get("/location-stats")
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data) == 1
    loc = data[0]
    assert loc["zone_count"] == 2
    assert len(loc["zones"]) == 2

    # pole1 stats
    pole1_info = loc["zones"][0]["poles"][0]
    assert pole1_info["camera_count"] == 2
    assert pole1_info["ticket_count"] == 3
    assert pole1_info["manual_review_count"] == 1

    pole2_info = loc["zones"][1]["poles"][0]
    assert pole2_info["camera_count"] == 1
    assert pole2_info["ticket_count"] == 3
    assert pole2_info["manual_review_count"] == 2

