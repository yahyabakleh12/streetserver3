import os
import base64
from datetime import datetime
from unittest.mock import patch
import types
import sys

from fastapi.testclient import TestClient
import pytest

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB
os.environ["DRAMATIQ_ALWAYS_EAGER"] = "true"
os.environ["DISABLE_TICKET_RETRY_WORKER"] = "true"


class DummyYOLO:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self


ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = DummyYOLO
sys.modules.setdefault("ultralytics", ultralytics_stub)

logic_stub = types.ModuleType("logic")
logic_stub.crop_and_save_car = lambda *args, **kwargs: ""
logic_stub.same_entired_car = lambda *args, **kwargs: False
logic_stub.exit_video_analyses = lambda *args, **kwargs: False
sys.modules.setdefault("logic", logic_stub)

utils_stub = types.ModuleType("utils")
utils_stub.is_same_car = lambda *args, **kwargs: False
sys.modules.setdefault("utils", utils_stub)

import network as _network_module


def _noop_ping(*args, **kwargs):
    return None


_network_module.ping_all_cameras = _noop_ping

from db import Base, engine, SessionLocal
from main import app, _process_pending_ticket_payloads
from models import Location, Zone, Pole, Camera, Spot, Ticket, PendingTicketPayload


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
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
        status=1,
    )
    session.add(spot)
    session.commit()
    ticket = Ticket(
        camera_id=cam.id,
        spot_number=1,
        plate_number="AAA",
        plate_code="1",
        plate_city="DXB",
        confidence=90,
        entry_time=datetime.utcnow(),
    )
    session.add(ticket)
    session.commit()
    ticket_id = ticket.id
    session.close()

    from PIL import Image
    snap_path = tmp_path / "snap.jpg"
    Image.new("RGB", (20, 20)).save(snap_path)
    snap_b64 = base64.b64encode(snap_path.read_bytes()).decode()

    payload = {
        "event": "E",
        "device": "D",
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "report_type": "R",
        "resolution_w": 10,
        "resolution_y": 10,
        "parking_area": f"{loc.code}{cam.api_code}",
        "index_number": 1,
        "occupancy": 0,
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
    return payload, ticket_id


def test_exit_spot_still_occupied(client, tmp_path):
    payload, ticket_id = setup_db(tmp_path)
    with patch("main.send_ticket_payload", return_value={"status": "success"}):
        resp = client.post("/post", json=payload)
    assert resp.status_code == 200
    assert resp.json()["message"] == "Entry processed"
    # Tasks run synchronously in tests, so no join needed
    session = SessionLocal()
    t = session.query(Ticket).get(ticket_id)
    spot = session.query(Spot).first()
    session.close()
    assert t.exit_time is not None
    assert spot.status == 0


def test_exit_closes_ticket(client, tmp_path):
    payload, ticket_id = setup_db(tmp_path)
    with patch("main.send_ticket_payload", return_value={"status": "success"}):
        resp = client.post("/post", json=payload)
    assert resp.status_code == 200
    assert resp.json()["message"] == "Entry processed"
    # Tasks run synchronously in tests, so no join needed
    session = SessionLocal()
    t = session.query(Ticket).get(ticket_id)
    spot = session.query(Spot).first()
    session.close()
    assert t.exit_time is not None
    assert t.exit_clip_path is not None
    assert os.path.isfile(t.exit_clip_path)
    assert spot.status == 0


def test_exit_snapshot_match(client, tmp_path):
    payload, ticket_id = setup_db(tmp_path)
    with patch("main.send_ticket_payload", return_value={"status": "success"}) as mock_send:
        resp = client.post("/post", json=payload)
        assert resp.status_code == 200
        assert resp.json()["message"] == "Entry processed"
        assert mock_send.called
    session = SessionLocal()
    t = session.query(Ticket).get(ticket_id)
    spot = session.query(Spot).first()
    pending = session.query(PendingTicketPayload).all()
    session.close()
    assert not pending
    assert t.exit_time is not None
    assert t.exit_clip_path is not None
    assert os.path.isfile(t.exit_clip_path)
    assert spot.status == 0


def test_exit_status_code_acknowledged(client, tmp_path):
    payload, ticket_id = setup_db(tmp_path)
    with patch("main.send_ticket_payload", return_value={"status": 200}):
        resp = client.post("/post", json=payload)
        assert resp.status_code == 200
        assert resp.json()["message"] == "Entry processed"

    session = SessionLocal()
    pending = session.query(PendingTicketPayload).all()
    ticket = session.query(Ticket).get(ticket_id)
    session.close()

    assert not pending
    assert ticket.exit_time is not None


def test_exit_non_acknowledged_response_does_not_enqueue(client, tmp_path):
    payload, ticket_id = setup_db(tmp_path)
    with patch("main.send_ticket_payload", return_value={"status": "error"}):
        resp = client.post("/post", json=payload)
        assert resp.status_code == 200
        assert resp.json()["message"] == "Entry processed"

    session = SessionLocal()
    pending = session.query(PendingTicketPayload).all()
    ticket = session.query(Ticket).get(ticket_id)
    session.close()

    assert not pending
    assert ticket.exit_time is not None


def test_exit_delivery_timeout_enqueues_retry(client, tmp_path):
    payload, ticket_id = setup_db(tmp_path)
    with patch("main.send_ticket_payload", side_effect=Exception("timeout")):
        resp = client.post("/post", json=payload)
        assert resp.status_code == 200
        assert resp.json()["message"] == "Entry processed"
    session = SessionLocal()
    pending = session.query(PendingTicketPayload).all()
    ticket = session.query(Ticket).get(ticket_id)
    session.close()
    assert len(pending) == 1
    queued_payload = pending[0]
    assert queued_payload.payload["ticket_key_id"] == ticket_id
    assert queued_payload.attempt_count == 1
    assert ticket.exit_time is not None


def test_exit_retry_success(client, tmp_path):
    payload, ticket_id = setup_db(tmp_path)
    with patch("main.send_ticket_payload", side_effect=Exception("timeout")):
        resp = client.post("/post", json=payload)
        assert resp.status_code == 200
        assert resp.json()["message"] == "Entry processed"

    session = SessionLocal()
    pending_before = session.query(PendingTicketPayload).count()
    session.close()
    assert pending_before == 1

    with patch("main.send_ticket_payload", return_value={"status": "success"}) as mock_retry:
        processed = _process_pending_ticket_payloads()
        assert processed == 1
        assert mock_retry.called

    session = SessionLocal()
    pending_after = session.query(PendingTicketPayload).count()
    ticket = session.query(Ticket).get(ticket_id)
    session.close()
    assert pending_after == 0
    assert ticket.exit_time is not None
