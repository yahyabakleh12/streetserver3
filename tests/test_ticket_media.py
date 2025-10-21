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

image_path = "test_image.jpg"
with open(image_path, "wb") as f:
    f.write(b"img")
video_path = "test_video.mp4"
with open(video_path, "wb") as f:
    f.write(b"vid")

ticket = Ticket(
    camera_id=cam.id,
    spot_number=1,
    plate_number="ABC",
    entry_time=datetime.utcnow(),
    entry_image_path=image_path,
    exit_clip_path=video_path,
)
session.add(ticket)
session.commit()
session.close()

ticket_id = ticket.id

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
    for path in [image_path, video_path]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

def test_get_ticket_image(client):
    resp = client.get(f"/tickets/{ticket_id}/image")
    assert resp.status_code == 200
    assert resp.content == b"img"

def test_get_ticket_video(client):
    resp = client.get(f"/tickets/{ticket_id}/video")
    assert resp.status_code == 200
    assert resp.content == b"vid"
