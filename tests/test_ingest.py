import base64
import importlib
import os
import sys
import types

import orjson
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text


class DummyYOLO:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self


ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = DummyYOLO
sys.modules.setdefault("ultralytics", ultralytics_stub)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    stored = {}
    original_db_module = sys.modules.pop("db", None)
    original_main_module = sys.modules.pop("main", None)
    original_database_url = os.environ.get("DATABASE_URL")
    original_post_queue_inline = os.environ.get("POST_QUEUE_INLINE")

    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'ingest.db'}"
    os.environ["POST_QUEUE_INLINE"] = "1"
    db_module = importlib.import_module("db")
    main_module = importlib.import_module("main")

    def fake_put_snapshot(key, body, content_type="image/jpeg"):
        stored["key"] = key
        stored["body"] = body
        stored["content_type"] = content_type
        return key

    class DummyCV2:
        IMREAD_COLOR = 1

        @staticmethod
        def imdecode(array, flag):
            return main_module.np.zeros((1, 1, 3), dtype=main_module.np.uint8)

        @staticmethod
        def imencode(ext, image):
            return True, main_module.np.frombuffer(b"jpeg-bytes", dtype=main_module.np.uint8)

    monkeypatch.setattr(main_module, "put_snapshot", fake_put_snapshot)
    monkeypatch.setattr(main_module, "cv2", DummyCV2)

    base = db_module.Base
    engine = db_module.engine

    base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS camera_reports (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts TIMESTAMP NOT NULL,
                  camera_id TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  occupancy INTEGER,
                  bbox TEXT,
                  image_key TEXT,
                  payload TEXT NOT NULL
                );
                """
            )
        )
        connection.execute(text("DELETE FROM camera_reports"))

    stored["engine"] = engine

    try:
        with TestClient(main_module.app) as c:
            yield c, stored
    finally:
        engine.dispose()
        sys.modules.pop("main", None)
        sys.modules.pop("db", None)
        if original_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = original_database_url
        if original_post_queue_inline is None:
            os.environ.pop("POST_QUEUE_INLINE", None)
        else:
            os.environ["POST_QUEUE_INLINE"] = original_post_queue_inline
        if original_db_module is not None:
            sys.modules["db"] = original_db_module
        if original_main_module is not None:
            sys.modules["main"] = original_main_module


def test_fast_ingest_flow(client):
    client_obj, stored = client

    payload = {
        "device": "CAM-123",
        "time": "2024-01-01T00:00:00Z",
        "occupancy": 1,
        "event": "entry",
        "snapshot": base64.b64encode(b"not-a-real-image").decode("ascii"),
    }

    response = client_obj.post(
        "/ingest",
        content=orjson.dumps(payload),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 200
    resp_json = response.json()
    assert resp_json["status"] == "ok"
    assert resp_json["image_key"] == stored["key"]
    assert stored["content_type"] == "image/jpeg"
    assert stored["body"] == b"jpeg-bytes"

    with stored["engine"].begin() as connection:
        row = connection.execute(
            text(
                "SELECT camera_id, occupancy, image_key, payload "
                "FROM camera_reports"
            )
        ).fetchone()

    assert row is not None
    mapping = row._mapping
    assert mapping["camera_id"] == payload["device"]
    assert mapping["occupancy"] == payload["occupancy"]
    assert mapping["image_key"] == resp_json["image_key"]

    stored_payload = orjson.loads(mapping["payload"])
    assert "snapshot" not in stored_payload
    assert stored_payload["device"] == payload["device"]
