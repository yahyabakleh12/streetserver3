import os
import base64
from datetime import datetime
from unittest.mock import patch
import types
import sys

import pytest
from fastapi.testclient import TestClient

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB
os.environ["POST_QUEUE_INLINE"] = "1"

# Provide a lightweight stub for the optional ultralytics dependency so that
# importing ``ocr_processor`` inside ``main`` doesn't require the real package
class DummyYOLO:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        pass

ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = DummyYOLO
sys.modules.setdefault("ultralytics", ultralytics_stub)

from db import Base, engine, SessionLocal
from main import app
from models import Location, Zone, Pole, Camera, Spot, Ticket


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)
    try:
        os.remove("test.db")
    except FileNotFoundError:
        pass


def setup_db(tmp_path):
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    loc = Location(
        name="Loc",
        code="LOC",
        portal_name="u",
        portal_password="p",
        ip_schema="ip",
        parkonic_api_token="tok",
        camera_user="user",
        camera_pass="pass",
    )
    session.add(loc)
    session.commit()
    zone = Zone(code="Z1", location_id=loc.id)
    session.add(zone)
    session.commit()
    pole = Pole(zone_id=zone.id, code="P1", location_id=loc.id, api_pole_id=1)
    session.add(pole)
    session.commit()
    cam = Camera(pole_id=pole.id, api_code="123", p_ip="127.0.0.1", portal_id=1)
    session.add(cam)
    session.commit()
    spot = Spot(
        camera_id=cam.id,
        spot_number=1,
        p1_x=0,
        p1_y=0,
        p2_x=10,
        p2_y=0,
        p3_x=10,
        p3_y=10,
        p4_x=0,
        p4_y=10,
    )
    session.add(spot)
    session.commit()
    session.close()

    from PIL import Image
    snap_path = tmp_path / "snap.jpg"
    Image.new("RGB", (20, 20)).save(snap_path)
    snap_b64 = base64.b64encode(snap_path.read_bytes()).decode()

    payload = {
        "event": "E",
        "device": "D",
        "time": datetime.utcnow().isoformat(),
        "report_type": "R",
        "resolution_w": 10,
        "resolution_y": 10,
        "parking_area": f"{loc.code}{cam.api_code}",
        "index_number": 1,
        "occupancy": 1,
        "duration": 1,
        "coordinate_x1": 0,
        "coordinate_y1": 0,
        "coordinate_x2": 1,
        "coordinate_y2": 0,
        "coordinate_x3": 1,
        "coordinate_y3": 1,
        "coordinate_x4": 0,
        "coordinate_y4": 1,
        "vehicle_frame_x1": 0,
        "vehicle_frame_y1": 0,
        "vehicle_frame_x2": 1,
        "vehicle_frame_y2": 1,
        "snapshot": snap_b64,
    }
    return payload


def test_entry_no_car_detected(client, tmp_path):
    payload = setup_db(tmp_path)
    with patch("main.spot_has_car", return_value=False):
        resp = client.post("/post", json=payload)
    assert resp.status_code == 200
    assert resp.json()["message"] == "Entry processed"
    # Tasks run synchronously in tests, so no join needed
    session = SessionLocal()
    tickets = session.query(Ticket).all()
    session.close()
    assert len(tickets) == 0


def test_entry_with_car(client, tmp_path):
    payload = setup_db(tmp_path)
    with patch("main.spot_has_car", return_value=True), \
         patch("main._process_plate_task") as mock_proc:
        resp = client.post("/post", json=payload)
    assert resp.status_code == 200
    assert resp.json()["message"] == "Entry processed"
    mock_proc.assert_called_once()
    session = SessionLocal()
    spot = session.query(Spot).first()
    session.close()
    assert spot.status == 1


def test_post_task_processed(client, tmp_path):
    payload = setup_db(tmp_path)
    with patch("main.spot_has_car", return_value=False), patch(
        "main._process_post_task"
    ) as mock_post:
        resp = client.post("/post", json=payload)
    assert resp.status_code == 200
    mock_post.assert_called_once()
