import os
import base64
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

# Set up test database before importing app
TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from db import Base, engine, SessionLocal
from main import app
import main
from models import Location, Zone, Pole, Camera, Ticket, ManualReview, User
from main import get_password_hash

# Create tables
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
    # Clean up DB file after tests
    Base.metadata.drop_all(bind=engine)
    try:
        os.remove("test.db")
    except FileNotFoundError:
        pass

@pytest.fixture()
def sample_review(tmp_path):
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
    cam = Camera(pole_id=pole.id, api_code="C1", p_ip="127.0.0.1", portal_id=1)
    session.add(cam)
    session.commit()
    ticket = Ticket(
        camera_id=cam.id,
        spot_number=1,
        plate_number="OLD",
        plate_code="1",
        plate_city="DXB",
        confidence=50,
        entry_time=datetime.utcnow(),
    )
    session.add(ticket)
    session.commit()
    ticket.image_base64 = base64.b64encode(b"imgdata").decode()
    session.commit()
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(b"data")
    review = ManualReview(
        camera_id=cam.id,
        spot_number=1,
        event_time=datetime.utcnow(),
        image_path=str(img_path),
        clip_path=None,
        ticket_id=ticket.id,
        plate_status="UNREAD",
        plate_image="plate.jpg",
        snapshot_folder="folder",
        review_status="PENDING",
    )
    session.add(review)
    session.commit()
    session.close()
    return review.id, ticket.id


@pytest.fixture()
def review_with_video(tmp_path):
    """Create a manual review that already has an associated clip."""
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
    cam = Camera(pole_id=pole.id, api_code="C1", p_ip="127.0.0.1", portal_id=1)
    session.add(cam)
    session.commit()
    ticket = Ticket(
        camera_id=cam.id,
        spot_number=1,
        plate_number="OLD",
        plate_code="1",
        plate_city="DXB",
        confidence=50,
        entry_time=datetime.utcnow(),
    )
    session.add(ticket)
    session.commit()
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"vid")
    review = ManualReview(
        camera_id=cam.id,
        spot_number=1,
        event_time=datetime.utcnow(),
        image_path=str(clip_path),
        clip_path=str(clip_path),
        ticket_id=ticket.id,
        plate_status="UNREAD",
        plate_image="plate.jpg",
        snapshot_folder="folder",
        review_status="PENDING",
    )
    session.add(review)
    session.commit()
    session.close()
    return review.id

from unittest.mock import patch


def test_correct_manual_review_success(client, sample_review):
    review_id, ticket_id = sample_review
    payload = {
        "plate_number": "NEW123",
        "plate_code": "90",
        "plate_city": "DXB",
        "confidence": 99,
    }
    with patch("api_client.park_in_request", return_value=None):
        response = client.post(f"/manual-reviews/{review_id}/correct", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "updated"}

    session = SessionLocal()
    ticket = session.query(Ticket).get(ticket_id)
    review = session.query(ManualReview).get(review_id)
    assert ticket.plate_number == "NEW123"
    assert ticket.plate_code == "90"
    assert ticket.plate_city == "DXB"
    assert ticket.confidence == 99
    assert review.review_status == "RESOLVED"
    assert review.plate_status == "READ"
    session.close()


def test_correct_manual_review_uses_ticket_image(client, sample_review):
    review_id, ticket_id = sample_review
    session = SessionLocal()
    ticket = session.query(Ticket).get(ticket_id)
    img_b64 = ticket.image_base64
    session.close()

    payload = {
        "plate_number": "IMG",
        "plate_code": "99",
        "plate_city": "DXB",
        "confidence": 70,
    }
    with patch("api_client.park_in_request", return_value=None):
        resp = client.post(f"/manual-reviews/{review_id}/correct", json=payload)
    assert resp.status_code == 200


def test_correct_manual_review_not_found(client):
    payload = {
        "plate_number": "X",
        "plate_code": "1",
        "plate_city": "DXB",
        "confidence": 80,
    }
    with patch("api_client.park_in_request", return_value=None):
        response = client.post("/manual-reviews/999/correct", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"] == "Review not found"


def test_correct_manual_review_validation_error(client, sample_review):
    review_id, _ = sample_review
    payload = {
        "plate_number": "NEW",
        # Missing plate_code
        "plate_city": "DXB",
        "confidence": 80,
    }
    response = client.post(f"/manual-reviews/{review_id}/correct", json=payload)
    assert response.status_code == 422


def test_correct_manual_review_with_exit(client, sample_review):
    review_id, ticket_id = sample_review
    session = SessionLocal()
    ticket = session.query(Ticket).get(ticket_id)
    ticket.exit_time = datetime.utcnow()
    ticket.parkonic_trip_id = 5
    session.commit()
    session.close()

    payload = {
        "plate_number": "EXIT",
        "plate_code": "77",
        "plate_city": "DXB",
        "confidence": 88,
    }
    with patch("api_client.park_in_request", return_value=None), \
         patch("api_client.park_out_request", return_value=None):
        resp = client.post(f"/manual-reviews/{review_id}/correct", json=payload)
    assert resp.status_code == 200


def test_external_manual_correction_open_ticket(sample_review):
    review_id, ticket_id = sample_review
    payload = {
        "review_id": review_id,
        "plate_number": "EXT",
        "plate_code": "11",
        "plate_city": "DXB",
        "image_base64": "img",
    }
    with TestClient(app) as c, \
         patch("api_client.park_in_request", return_value=None), \
         patch("api_client.park_out_request2", return_value=None):
        resp = c.post("/external-corrections", json=payload)
    assert resp.status_code == 200

    session = SessionLocal()
    ticket = session.query(Ticket).get(ticket_id)
    review = session.query(ManualReview).get(review_id)
    assert ticket.plate_number == "EXT"
    assert review.review_status == "RESOLVED"
    session.close()


def test_external_manual_correction_with_exit(sample_review):
    review_id, ticket_id = sample_review
    session = SessionLocal()
    ticket = session.query(Ticket).get(ticket_id)
    ticket.exit_time = datetime.utcnow()
    ticket.parkonic_trip_id = 5
    session.commit()
    session.close()

    payload = {
        "review_id": review_id,
        "plate_number": "EXIT",
        "plate_code": "77",
        "plate_city": "DXB",
        "image_base64": "img",
    }
    with TestClient(app) as c, \
         patch("api_client.park_in_request", return_value=None), \
         patch("api_client.park_out_request2", return_value=None):
        resp = c.post("/external-corrections", json=payload)
    assert resp.status_code == 200


def test_external_manual_correction_with_exit_no_trip(sample_review):
    review_id, ticket_id = sample_review
    session = SessionLocal()
    ticket = session.query(Ticket).get(ticket_id)
    ticket.exit_time = datetime.utcnow()
    ticket.parkonic_trip_id = None
    session.commit()
    session.close()

    payload = {
        "review_id": review_id,
        "plate_number": "EXIT",
        "plate_code": "77",
        "plate_city": "DXB",
        "image_base64": "img",
    }
    with TestClient(app) as c, \
         patch("api_client.park_in_request", return_value={"trip_id": 123}), \
         patch("api_client.park_out_request2", return_value=None):
        resp = c.post("/external-corrections", json=payload)
    assert resp.status_code == 200
    session = SessionLocal()
    ticket = session.query(Ticket).get(ticket_id)
    assert ticket.parkonic_trip_id is None
    session.close()


def test_external_manual_correction_reopen_previous_ticket(tmp_path):
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
    cam = Camera(pole_id=pole.id, api_code="C1", p_ip="127.0.0.1", portal_id=1)
    session.add(cam)
    session.commit()
    # Closed previous ticket
    prev_ticket = Ticket(
        camera_id=cam.id,
        spot_number=1,
        plate_number="AAA",
        plate_code="77",
        plate_city="DXB",
        confidence=80,
        entry_time=datetime.utcnow(),
        exit_time=datetime.utcnow(),
    )
    session.add(prev_ticket)
    session.commit()
    prev_id = prev_ticket.id
    # New ticket linked to review
    new_ticket = Ticket(
        camera_id=cam.id,
        spot_number=1,
        plate_number="TEMP",
        plate_code="77",
        plate_city="DXB",
        confidence=50,
        entry_time=datetime.utcnow(),
    )
    session.add(new_ticket)
    session.commit()
    new_id = new_ticket.id
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(b"data")
    review = ManualReview(
        camera_id=cam.id,
        spot_number=1,
        event_time=datetime.utcnow(),
        image_path=str(img_path),
        clip_path=None,
        ticket_id=new_id,
        plate_status="UNREAD",
        plate_image="plate.jpg",
        snapshot_folder="folder",
        review_status="PENDING",
    )
    session.add(review)
    session.commit()
    review_id = review.id
    session.close()

    payload = {
        "review_id": review_id,
        "plate_number": "AAA",
        "plate_code": "77",
        "plate_city": "DXB",
        "image_base64": "img",
    }
    with TestClient(app) as c, \
         patch("api_client.park_in_request", return_value=None), \
         patch("api_client.park_out_request2", return_value=None):
        resp = c.post("/external-corrections", json=payload)
    assert resp.status_code == 200

    session = SessionLocal()
    reopened = session.query(Ticket).get(prev_id)
    deleted = session.query(Ticket).get(new_id)
    review_db = session.query(ManualReview).get(review_id)
    assert reopened.exit_time is None
    assert review_db.ticket_id == prev_id
    assert deleted is None
    session.close()


def test_send_pending_review_videos(client, review_with_video):
    review_id = review_with_video
    with patch("api_client.send_review_video", return_value=None) as mock_send:
        resp = client.post("/manual-reviews/send-videos")
    assert resp.status_code == 200
    assert resp.json()["dispatched"] == 1
    assert mock_send.called
    assert mock_send.call_args.kwargs["review_id"] == review_id

@pytest.fixture()
def multi_camera_reviews(tmp_path):
    """Create pending reviews with clips for two cameras."""
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

    review_ids = {}
    for i in range(2):
        cam = Camera(pole_id=pole.id, api_code=f"C{i+1}", p_ip="127.0.0.1", portal_id=i + 1)
        session.add(cam)
        session.commit()
        ticket = Ticket(
            camera_id=cam.id,
            spot_number=1,
            plate_number="OLD",
            plate_code="1",
            plate_city="DXB",
            confidence=50,
            entry_time=datetime.utcnow(),
        )
        session.add(ticket)
        session.commit()
        for j in range(2):
            clip_path = tmp_path / f"clip_{i}_{j}.mp4"
            clip_path.write_bytes(b"vid")
            review = ManualReview(
                camera_id=cam.id,
                spot_number=1,
                event_time=datetime.utcnow(),
                image_path=str(clip_path),
                clip_path=str(clip_path),
                ticket_id=ticket.id,
                plate_status="UNREAD",
                plate_image="plate.jpg",
                snapshot_folder="folder",
                review_status="PENDING",
            )
            session.add(review)
            session.commit()
            if j == 0:
                review_ids[cam.id] = review.id
    session.close()
    return review_ids


@pytest.fixture()
def missing_first_clip_reviews(tmp_path):
    """Create two cameras where the first review lacks a video file."""
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

    review_ids = {}
    for i in range(2):
        cam = Camera(pole_id=pole.id, api_code=f"C{i+1}", p_ip="127.0.0.1", portal_id=i + 1)
        session.add(cam)
        session.commit()
        ticket = Ticket(
            camera_id=cam.id,
            spot_number=1,
            plate_number="OLD",
            plate_code="1",
            plate_city="DXB",
            confidence=50,
            entry_time=datetime.utcnow(),
        )
        session.add(ticket)
        session.commit()
        # First review has missing clip
        missing_path = tmp_path / f"missing_{i}.mp4"
        review = ManualReview(
            camera_id=cam.id,
            spot_number=1,
            event_time=datetime.utcnow(),
            image_path=str(missing_path),
            clip_path=str(missing_path),
            ticket_id=ticket.id,
            plate_status="UNREAD",
            plate_image="plate.jpg",
            snapshot_folder="folder",
            review_status="PENDING",
        )
        session.add(review)
        session.commit()

        # Second review with existing clip
        clip_path = tmp_path / f"clip_{i}.mp4"
        clip_path.write_bytes(b"vid")
        review = ManualReview(
            camera_id=cam.id,
            spot_number=1,
            event_time=datetime.utcnow(),
            image_path=str(clip_path),
            clip_path=str(clip_path),
            ticket_id=ticket.id,
            plate_status="UNREAD",
            plate_image="plate.jpg",
            snapshot_folder="folder",
            review_status="PENDING",
        )
        session.add(review)
        session.commit()
        review_ids[cam.id] = review.id
    session.close()
    return review_ids


@pytest.fixture()
def cleanup_review_set(tmp_path):
    """Create reviews with and without clip files for cleanup testing."""
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
    cam = Camera(pole_id=pole.id, api_code="C1", p_ip="127.0.0.1", portal_id=1)
    session.add(cam)
    session.commit()
    ticket = Ticket(
        camera_id=cam.id,
        spot_number=1,
        plate_number="OLD",
        plate_code="1",
        plate_city="DXB",
        confidence=50,
        entry_time=datetime.utcnow(),
    )
    session.add(ticket)
    session.commit()

    # Review without clip path
    review1 = ManualReview(
        camera_id=cam.id,
        spot_number=1,
        event_time=datetime.utcnow(),
        image_path="img1.jpg",
        clip_path=None,
        ticket_id=ticket.id,
        plate_status="UNREAD",
        plate_image="plate.jpg",
        snapshot_folder="folder",
        review_status="PENDING",
    )
    session.add(review1)
    session.commit()

    # Review with non-existent clip file
    missing_path = tmp_path / "missing.mp4"
    review2 = ManualReview(
        camera_id=cam.id,
        spot_number=1,
        event_time=datetime.utcnow(),
        image_path=str(missing_path),
        clip_path=str(missing_path),
        ticket_id=ticket.id,
        plate_status="UNREAD",
        plate_image="plate.jpg",
        snapshot_folder="folder",
        review_status="PENDING",
    )
    session.add(review2)
    session.commit()

    # Review with existing clip file
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"vid")
    review3 = ManualReview(
        camera_id=cam.id,
        spot_number=1,
        event_time=datetime.utcnow(),
        image_path=str(clip_path),
        clip_path=str(clip_path),
        ticket_id=ticket.id,
        plate_status="UNREAD",
        plate_image="plate.jpg",
        snapshot_folder="folder",
        review_status="PENDING",
    )
    session.add(review3)
    session.commit()
    session.close()
    return review1.id, review2.id, review3.id


def test_send_one_review_video_per_camera(client, multi_camera_reviews):
    expected = set(multi_camera_reviews.values())
    with patch("api_client.send_review_video", return_value=None) as mock_send:
        resp = client.post("/manual-reviews/send-one-video")
    assert resp.status_code == 200
    assert resp.json()["dispatched"] == len(expected)
    sent_ids = {call.kwargs["review_id"] for call in mock_send.call_args_list}
    assert sent_ids == expected


def test_send_one_review_video_per_camera_skips_missing(client, missing_first_clip_reviews):
    expected = set(missing_first_clip_reviews.values())
    with patch("api_client.send_review_video", return_value=None) as mock_send:
        resp = client.post("/manual-reviews/send-one-video")
    assert resp.status_code == 200
    assert resp.json()["dispatched"] == len(expected)
    sent_ids = {call.kwargs["review_id"] for call in mock_send.call_args_list}
    assert sent_ids == expected


def test_delete_reviews_missing_videos(client, cleanup_review_set):
    rid1, rid2, rid3 = cleanup_review_set
    resp = client.post("/manual-reviews/delete-missing-videos")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 2
    session = SessionLocal()
    assert session.query(ManualReview).get(rid1) is None
    assert session.query(ManualReview).get(rid2) is None
    assert session.query(ManualReview).get(rid3) is not None
    session.close()


def test_get_review_snapshot_safe_filename(client, sample_review, tmp_path, monkeypatch):
    review_id, _ = sample_review
    monkeypatch.setattr(main, "SNAPSHOTS_DIR", str(tmp_path))
    folder = tmp_path / "folder"
    folder.mkdir(parents=True, exist_ok=True)
    snap = folder / "safe.jpg"
    snap.write_bytes(b"ok")
    resp = client.get(f"/manual-reviews/{review_id}/snapshots/{snap.name}")
    assert resp.status_code == 200
    assert resp.content == b"ok"


def test_get_review_snapshot_rejects_malicious_filename(client, sample_review, tmp_path, monkeypatch):
    review_id, _ = sample_review
    monkeypatch.setattr(main, "SNAPSHOTS_DIR", str(tmp_path))
    folder = tmp_path / "folder"
    folder.mkdir(parents=True, exist_ok=True)
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret")
    resp = client.get(
        f"/manual-reviews/{review_id}/snapshots/../../{outside.name}"
    )
    assert resp.status_code == 404
