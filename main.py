# main.py

import os
import io
import re
import base64
import binascii
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import numpy as np
import time
from pathlib import Path
from typing import Any
import orjson
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
from network import (
    send_request_with_retry,
    send_request_with_retry_async,
    session as network_session,
)
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Depends,
    UploadFile,
    File,
)
from fastapi.responses import ORJSONResponse, JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import requests
from sqlalchemy import text, asc, desc, func
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.orm import joinedload
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
import random
from PIL import Image, ImageDraw
from db import SessionLocal, engine, Base
from models import (
    Report,
    Ticket,
    Camera,
    Pole,
    Location,
    Spot,
    CropZone,
    ManualReview,
    ClipRequest,
    Zone,
    User,
    Role,
    Permission,
    PendingTicketPayload,
)
from ocr_processor import (
    process_plate_and_issue_ticket,
    spot_has_car,
    exit_decision_from_frame,
)
import ocr_processor
from camera_clip import (
    request_camera_clip,
    is_valid_mp4,
    fetch_camera_frame,
)
from logger import logger
from utils import is_same_car
from network import ping_all_cameras
from config import API_LOCATION_ID, OCR_TOKEN
from image_enhancer import enhance_image_array
from storage import put_snapshot

from pydantic import BaseModel
from logic import exit_video_analyses

app = FastAPI(default_response_class=ORJSONResponse)




# Shared thread pool for blocking tasks
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("MAX_WORKERS", "32")))

# Shared HTTP session for outgoing requests
http_session = requests.Session()


@dataclass(slots=True)
class _PostTask:
    payload: dict[str, Any]
    raw_body: bytes
    ts: str


_POST_TASK_QUEUE: "queue.Queue[_PostTask | object]" = queue.Queue()
_POST_QUEUE_SENTINEL = object()
_post_worker_stop_event = threading.Event()
_post_worker_thread: threading.Thread | None = None


def _should_process_post_inline() -> bool:
    flag = os.environ.get("POST_QUEUE_INLINE", "")
    return flag.lower() in {"1", "true", "yes"}


def _enqueue_post_task(payload: dict[str, Any], raw_body: bytes, ts: str) -> None:
    if _should_process_post_inline():
        _process_post_task(payload, raw_body, ts)
        return
    _POST_TASK_QUEUE.put(_PostTask(payload=payload, raw_body=raw_body, ts=ts))


def _post_worker_loop() -> None:
    while True:
        try:
            task = _POST_TASK_QUEUE.get(timeout=1)
        except queue.Empty:
            if _post_worker_stop_event.is_set():
                break
            continue
        if task is _POST_QUEUE_SENTINEL:
            _POST_TASK_QUEUE.task_done()
            break
        assert isinstance(task, _PostTask)
        try:
            _process_post_task(task.payload, task.raw_body, task.ts)
        except Exception:
            logger.error("Background post task failed", exc_info=True)
        finally:
            _POST_TASK_QUEUE.task_done()


def _start_post_worker() -> None:
    global _post_worker_thread
    if _should_process_post_inline():
        return
    if _post_worker_thread and _post_worker_thread.is_alive():
        return
    _post_worker_stop_event.clear()
    _post_worker_thread = threading.Thread(
        target=_post_worker_loop,
        name="post-worker",
        daemon=True,
    )
    _post_worker_thread.start()


def _stop_post_worker() -> None:
    global _post_worker_thread
    if _should_process_post_inline():
        return
    _post_worker_stop_event.set()
    if _post_worker_thread and _post_worker_thread.is_alive():
        _POST_TASK_QUEUE.put(_POST_QUEUE_SENTINEL)
        _post_worker_thread.join(timeout=5)
    _post_worker_thread = None

TICKET_RETRY_INTERVAL = int(os.environ.get("TICKET_RETRY_INTERVAL", "60"))
TICKET_RETRY_BATCH_SIZE = int(os.environ.get("TICKET_RETRY_BATCH_SIZE", "20"))
_ticket_retry_stop_event = threading.Event()
_ticket_retry_thread = None


async def run_in_executor(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(EXECUTOR, func, *args)


# 1. Determine which origins are allowed to access the API.
#    By default a couple of development IPs are whitelisted, but this can be
#    overridden via the ``CORS_ORIGINS`` environment variable.  Use ``*`` to
#    allow any origin, or provide a comma separated list of hosts.
cors_env = os.environ.get("CORS_ORIGINS")

# 2. Add the CORS middleware *before* you include any routers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚙️ Allowed origins
    allow_credentials=True,  # ⚙️ Allow cookies, Authorization headers
    allow_methods=["*"],  # ⚙️ Allowed HTTP methods (GET, POST, ...)
    allow_headers=["*"],  # ⚙️ Allowed HTTP headers (Content-Type, Authorization, ...)
    expose_headers=["*"],  # (optional) headers you want JS to read
    max_age=3600,  # (optional) how long the results of a preflight request can be cached
)

# Directories for saving raw requests and snapshots
SNAPSHOTS_DIR = "snapshots"
RAW_REQUEST_DIR = os.path.join(SNAPSHOTS_DIR, "raw_request")
SPOT_LAST_DIR = "spot_last"  # where we keep the "last main_crop" per (camera, spot)
REPORTS_JSON_DIR = "reports_json"  # directory for storing JSON reports
OUTPUT_DIR = "check_similarity"
AN_VIDEO_OUTPUT_DIR = "AN_VIDEO_OUTPUT_DIR"
CAR_CROPS = "car_crops"
os.makedirs(CAR_CROPS, exist_ok=True)
os.makedirs(AN_VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_REQUEST_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(SPOT_LAST_DIR, exist_ok=True)
os.makedirs(REPORTS_JSON_DIR, exist_ok=True)
IGNORED_ENTRY_DIR = os.path.join(SNAPSHOTS_DIR, "ignored_entry")
IGNORED_EXIT_DIR = os.path.join(SNAPSHOTS_DIR, "ignored_exit")
os.makedirs(IGNORED_ENTRY_DIR, exist_ok=True)
os.makedirs(IGNORED_EXIT_DIR, exist_ok=True)
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    Base.metadata.create_all(bind=engine)
    _start_post_worker()
    if os.environ.get("DISABLE_TICKET_RETRY_WORKER", "").lower() not in {"1", "true", "yes"}:
        _start_ticket_retry_worker()


@app.on_event("shutdown")
def _shutdown() -> None:
    _stop_post_worker()
    _stop_ticket_retry_worker()
    http_session.close()
    network_session.close()

# ---------------------------------------------------------------------------
# Helper to stream files safely
def stream_file(path: str) -> StreamingResponse:
    """Return a StreamingResponse that logs if the client disconnects."""

    filename = os.path.basename(path)

    def _iterfile():
        try:
            with open(path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk
        except GeneratorExit:
            # Happens when the client disconnects before download completes
            logger.info("Client disconnected while streaming %s", path)
            raise

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(
        _iterfile(), headers=headers, media_type="application/octet-stream"
    )


# Determine camera_id to route tasks to the appropriate Celery shard
def _get_camera_id_from_payload(payload: dict) -> int:
    m = re.match(r"^([A-Za-z]+)(\d+)$", payload["parking_area"])
    if not m:
        raise HTTPException(
            status_code=400,
            detail="Invalid parking_area format (expected letters+digits, e.g. 'NAD95')",
        )
    location_code, api_code = m.group(1), m.group(2)
    db = SessionLocal()
    try:
        stmt = text(
            "SELECT c.id FROM cameras AS c "
            "JOIN poles AS p ON c.pole_id = p.id "
            "JOIN zones AS z ON p.zone_id = z.id "
            "JOIN locations AS l ON p.location_id = l.id "
            "WHERE l.code = :loc_code AND c.api_code = :api_code "
            "LIMIT 1"
        )
        row = db.execute(
            stmt, {"loc_code": location_code, "api_code": api_code}
        ).fetchone()
    finally:
        db.close()


@app.post("/echo")
async def echo(request: Request):
    """Echo back the request payload."""

    try:
        return await request.json()
    except Exception:
        data = await request.body()
        return {"data": data.decode()}


    if row is None:
        raise HTTPException(
            status_code=400, detail="No camera found for that parking_area"
        )
    return row[0]


def _parse_event_timestamp(value) -> datetime:
    """Return a timezone-naive UTC ``datetime`` for the provided value."""

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None)
        except (OverflowError, OSError, ValueError):
            return datetime.utcnow()

    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            if candidate.endswith("Z"):
                candidate = candidate[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                pass
            else:
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                return parsed

    return datetime.utcnow()


def _extract_bbox(payload: dict[str, Any]) -> list[list[int]] | None:
    point_fields = [
        ("coordinate_x1", "coordinate_y1"),
        ("coordinate_x2", "coordinate_y2"),
        ("coordinate_x3", "coordinate_y3"),
        ("coordinate_x4", "coordinate_y4"),
    ]
    points: list[list[int]] = []
    for x_key, y_key in point_fields:
        x_val = payload.get(x_key)
        y_val = payload.get(y_key)
        if x_val is None or y_val is None:
            return None
        try:
            points.append([int(x_val), int(y_val)])
        except (TypeError, ValueError):
            return None
    return points


def _store_snapshot_from_payload(payload: dict[str, Any], *, require_snapshot: bool) -> str | None:
    snapshot_b64 = payload.get("snapshot")
    if not isinstance(snapshot_b64, str) or not snapshot_b64.strip():
        if require_snapshot:
            raise HTTPException(status_code=422, detail="Missing snapshot field")
        logger.warning("Received payload without snapshot; skipping image persistence")
        return None

    snapshot_b64 = snapshot_b64.strip()
    if snapshot_b64.startswith("data:") and "," in snapshot_b64:
        snapshot_b64 = snapshot_b64.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(snapshot_b64, validate=True)
    except (binascii.Error, ValueError):
        if require_snapshot:
            raise HTTPException(status_code=400, detail="Snapshot must be base64 encoded")
        logger.warning("Invalid base64 snapshot received; skipping image persistence")
        return None

    if not raw_bytes:
        if require_snapshot:
            raise HTTPException(status_code=400, detail="Snapshot image is empty")
        logger.warning("Empty snapshot received; skipping image persistence")
        return None

    image_bytes = raw_bytes
    if cv2 is not None:  # pragma: no branch - optional dependency
        np_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if frame is None:
            if require_snapshot:
                raise HTTPException(status_code=400, detail="Snapshot is not a valid image")
            logger.warning("Snapshot could not be decoded; storing raw bytes")
        else:
            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                if require_snapshot:
                    raise HTTPException(status_code=500, detail="Failed to encode snapshot")
                logger.warning("Failed to encode snapshot; storing raw bytes")
            else:
                image_bytes = encoded.tobytes()
    elif require_snapshot:
        logger.warning("OpenCV unavailable; storing raw snapshot bytes without re-encoding")

    image_key = (
        f"camera-reports/{datetime.utcnow().strftime('%Y/%m/%d')}/{uuid.uuid4().hex}.jpg"
    )

    try:
        put_snapshot(image_key, image_bytes, content_type="image/jpeg")
    except Exception:  # pragma: no cover - external dependency failure
        logger.exception("Failed to upload snapshot to object storage")
        if require_snapshot:
            raise HTTPException(status_code=502, detail="Failed to store snapshot")
        return None

    return image_key


def _persist_camera_report(payload: dict[str, Any], *, require_snapshot: bool = True) -> str | None:
    image_key = _store_snapshot_from_payload(payload, require_snapshot=require_snapshot)

    event_ts = _parse_event_timestamp(payload.get("time") or payload.get("ts"))
    camera_identifier = (
        payload.get("device")
        or payload.get("camera_id")
        or payload.get("parking_area")
    )
    if not camera_identifier:
        raise HTTPException(status_code=422, detail="Missing camera identifier")

    occupancy_raw = payload.get("occupancy")
    try:
        occupancy_value = int(occupancy_raw) if occupancy_raw is not None else None
    except (TypeError, ValueError):
        occupancy_value = None

    event_type = payload.get("event") or payload.get("report_type") or "ingest"
    bbox = _extract_bbox(payload)

    payload_to_store = dict(payload)
    payload_to_store.pop("snapshot", None)

    session = SessionLocal()
    try:
        session.execute(
            text(
                """
                INSERT INTO camera_reports (ts, camera_id, event_type, occupancy, bbox, image_key, payload)
                VALUES (:ts, :camera_id, :event_type, :occupancy, :bbox, :image_key, :payload)
                """
            ),
            {
                "ts": event_ts,
                "camera_id": camera_identifier,
                "event_type": event_type,
                "occupancy": occupancy_value,
                "bbox": (
                    orjson.dumps(bbox).decode("utf-8") if bbox is not None else None
                ),
                "image_key": image_key,
                "payload": orjson.dumps(payload_to_store).decode("utf-8"),
            },
        )
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Failed to persist camera report")
        raise HTTPException(status_code=500, detail="Failed to persist camera report")
    finally:
        session.close()

    return image_key


@app.post("/ingest", response_class=ORJSONResponse)
async def ingest_camera_report(request: Request):
    """Fast ingest path that stores snapshots in object storage and the database."""

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")

    try:
        payload = orjson.loads(body)
    except orjson.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON payload must be an object")

    image_key = _persist_camera_report(payload, require_snapshot=True)

    return {"status": "ok", "image_key": image_key}


# ── Authentication setup ───────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "changeme")
ALGORITHM = "HS256"
# Extend token validity to 24 hours so users stay logged in longer
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

try:
    _bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    _bcrypt_context.hash("probe")
except Exception:
    logger.warning(
        "bcrypt backend unavailable; falling back to pbkdf2_sha256 for password hashing"
    )
    pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
else:
    pwd_context = _bcrypt_context
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
_ping_all_cameras_thread = threading.Thread(target=ping_all_cameras, daemon=True)
_ping_all_cameras_thread.start()


class LocationCreate(BaseModel):
    name: str
    code: str
    portal_name: str
    portal_password: str
    ip_schema: str
    parkonic_api_token: str | None = None
    camera_user: str | None = None
    camera_pass: str | None = None


class PoleCreate(BaseModel):
    zone_id: int
    name: str | None = None
    code: str
    location_id: int
    number_of_cameras: int | None = 0
    server: str | None = None
    router: str | None = None
    router_ip: str | None = None
    router_vpn_ip: str | None = None


class CameraCreate(BaseModel):
    pole_id: int
    name: str | None = None
    api_code: str
    p_ip: str
    number_of_parking: int | None = 0
    vpn_ip: str | None = None
    portal_id: int | None = None


class ManualCorrection(BaseModel):
    plate_number: str
    plate_code: str
    plate_city: str
    confidence: int


class ZoneCreate(BaseModel):
    code: str
    location_id: int
    parameters: dict | None = None


class LocationUpdate(BaseModel):
    name: str | None = None
    code: str | None = None
    portal_name: str | None = None
    portal_password: str | None = None
    ip_schema: str | None = None
    parkonic_api_token: str | None = None
    camera_user: str | None = None
    camera_pass: str | None = None
    parameters: dict | None = None


class PoleUpdate(BaseModel):
    zone_id: int | None = None
    name: str | None = None
    code: str | None = None
    location_id: int | None = None
    number_of_cameras: int | None = None
    server: str | None = None
    router: str | None = None
    router_ip: str | None = None
    router_vpn_ip: str | None = None
    location_coordinates: str | None = None


class CameraUpdate(BaseModel):
    pole_id: int | None = None
    name: str | None = None
    api_code: str | None = None
    p_ip: str | None = None
    number_of_parking: int | None = None
    vpn_ip: str | None = None
    portal_id: int | None = None


class ZoneUpdate(BaseModel):
    code: str | None = None
    location_id: int | None = None
    parameters: dict | None = None


class TicketUpdate(BaseModel):
    camera_id: int | None = None
    spot_number: int | None = None
    plate_number: str | None = None
    plate_code: str | None = None
    plate_city: str | None = None
    confidence: int | None = None
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    parkonic_trip_id: int | None = None
    image_base64: str | None = None
    entry_image_path: str | None = None
    exit_clip_path: str | None = None


class ReportUpdate(BaseModel):
    camera_id: int | None = None
    event: str | None = None
    report_type: str | None = None
    timestamp: datetime | None = None
    payload: dict | None = None


class ManualReviewUpdate(BaseModel):
    review_status: str | None = None


class ClipRequestCreate(BaseModel):
    camera_id: int
    start: datetime
    end: datetime


class UserCreate(BaseModel):
    username: str
    password: str
    role_ids: list[int] = []


class UserUpdate(BaseModel):
    username: str | None = None
    password: str | None = None
    role_ids: list[int] | None = None


class RoleCreate(BaseModel):
    name: str
    description: str | None = None
    permission_ids: list[int] = []


class RoleUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    permission_ids: list[int] | None = None


class PermissionCreate(BaseModel):
    name: str
    description: str | None = None


class PermissionUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class SpotCreate(BaseModel):
    camera_id: int
    spot_number: int
    p1_x: int
    p1_y: int
    p2_x: int
    p2_y: int
    p3_x: int
    p3_y: int
    p4_x: int
    p4_y: int
    status: int = 0


class SpotUpdate(BaseModel):
    spot_number: int | None = None
    p1_x: int | None = None
    p1_y: int | None = None
    p2_x: int | None = None
    p2_y: int | None = None
    p3_x: int | None = None
    p3_y: int | None = None
    p4_x: int | None = None
    p4_y: int | None = None
    status: int | None = None


class CropZoneCreate(BaseModel):
    camera_id: int
    points: list[dict]


class CropZoneUpdate(BaseModel):
    points: list[dict] | None = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire, "jti": str(uuid.uuid4())})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    db = SessionLocal()
    try:
        user = _retry_operation(
            lambda s: (
                s.query(User)
                .options(joinedload(User.roles).joinedload(Role.permissions))
                .filter(User.username == username)
                .first()
            ),
            db,
        )
    finally:
        db.close()
    if user is None:
        raise credentials_exception
    return user

def send_ticket_payload(
    token: str,
    access_point_id: int,
    number: str,
    code: str,
    city: str,
    status: str,
    entry_time: str,
    exit_time: str,
    entry_pic_base64: str,
    car_pic_base64: str,
    exit_video_path: str,
    ticket_key_id:str,
    spot_number:str,
    
    url: str = "http://10.11.5.103:18001/ticket",
):
    #######################################################################
    # payload = {
    #     # "token": token,
    #     # "access_point_id": access_point_id,
    #     # "number": number,
    #     # "code": code,
    #     # "city": city,
    #     # "spot_number": spot_number,
    #     "trip_p_id": trip_p_id,
    #     # "ticket_key_id": ticket_key_id,
    #     # "status": status,
    #     # "entry_time": entry_time,
    #     # "exit_time": exit_time,
    #     # "entry_pic_base64": entry_pic_base64,
    #     # "car_pic_base64": car_pic_base64,
    #     # "exit_video_path": exit_video_path
    # }
    #######################################################################
    """Send a complete parking ticket payload to the external API."""
    payload = {
        "token": token,
        "access_point_id": access_point_id,
        "number": number,
        "code": code,
        "city": city,
        "status": status,
        "entry_time": entry_time,
        "exit_time":exit_time,
        "entry_pic_base64": entry_pic_base64,
        "car_pic_base64": car_pic_base64,
        "exit_video_path": exit_video_path,
        "ticket_key_id":ticket_key_id,
        "spot_number":spot_number,
        "trip_p_id":0
       
    }
    log_payload = {
        k: ("<omitted>" if k in {"entry_pic_base64", "car_pic_base64"} else v)
        for k, v in payload.items()
    }
    logger.info("[SEND-TICKET] Sending payload: %s", log_payload)
    resp = send_request_with_retry(url, payload)
    if isinstance(resp, str):
        try:
            resp = json.loads(resp)
        except Exception:
            logger.error("[SEND-TICKET] JSON decode failed", exc_info=True)
            resp = {}
    return resp


def _stringify_response(resp: object) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    try:
        return json.dumps(resp)
    except Exception:
        return str(resp)


def _is_ticket_delivery_acknowledged(response: object) -> bool:
    if response is None:
        return False
    parsed = response
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            return False
    if not isinstance(parsed, dict):
        return False
    status = parsed.get("status")
    if isinstance(status, bool):
        if status:
            return True
    elif isinstance(status, (int, float)):
        if int(status) == 200:
            return True
    elif isinstance(status, str):
        lowered = status.lower()
        if lowered in {"success", "ok", "acknowledged"}:
            return True
        if status.isdigit() and int(status) == 200:
            return True
    success = parsed.get("success")
    if isinstance(success, bool):
        if success:
            return True
    code = parsed.get("code")
    if isinstance(code, int) and code == 200:
        return True
    if isinstance(code, str) and code.isdigit() and int(code) == 200:
        return True
    message = parsed.get("message")
    if isinstance(message, str) and any(
        token in message.lower() for token in ("success", "acknowledged", "ok")
    ):
        return True
    return False


def _enqueue_pending_ticket_payload(
    session,
    *,
    ticket_id: int | None,
    payload_kwargs: dict,
    error_message: str | None = None,
):
    pending = PendingTicketPayload(
        ticket_id=ticket_id,
        payload=payload_kwargs,
        attempt_count=1,
        last_attempt_at=datetime.utcnow(),
        last_error=error_message,
    )
    session.add(pending)
    return pending


def _process_pending_ticket_payloads(*, batch_size: int | None = None) -> int:
    """Attempt to redeliver queued ticket payloads.

    Returns the number of payloads that were successfully acknowledged.
    """

    db = SessionLocal()
    processed = 0
    try:
        # Ordering by ``created_at`` forces MySQL to sort the entire table when no
        # supporting index exists which can quickly exhaust the server's sort
        # buffer.  Ordering by the primary key keeps the results chronological for
        # auto-increment ids while allowing the query to leverage the existing
        # index.
        query = db.query(PendingTicketPayload).order_by(
            PendingTicketPayload.id.asc()
        )
        if batch_size:
            query = query.limit(batch_size)
        pendings = query.all()
        for pending in pendings:
            payload_kwargs = pending.payload or {}
            if not isinstance(payload_kwargs, dict):
                logger.error(
                    "Pending ticket payload id=%s has invalid payload; discarding",
                    pending.id,
                )
                db.delete(pending)
                _retry_commit(pending, db)
                continue
            try:
                response = send_ticket_payload(**payload_kwargs)
            except Exception as exc:
                pending.attempt_count = (pending.attempt_count or 0) + 1
                pending.last_attempt_at = datetime.utcnow()
                pending.last_error = str(exc)
                logger.error(
                    "Retry delivery failed for pending payload id=%s",
                    pending.id,
                    exc_info=True,
                )
                _retry_commit(pending, db)
                continue
            acknowledged = _is_ticket_delivery_acknowledged(response)
            pending.attempt_count = (pending.attempt_count or 0) + 1
            pending.last_attempt_at = datetime.utcnow()
            if acknowledged:
                logger.info(
                    "Pending ticket payload id=%s acknowledged; removing from queue",
                    pending.id,
                )
                db.delete(pending)
                _retry_commit(pending, db)
                processed += 1
            else:
                pending.last_error = _stringify_response(response)
                logger.warning(
                    "Pending ticket payload id=%s not acknowledged; removing from queue",
                    pending.id,
                )
                db.delete(pending)
                _retry_commit(pending, db)
        return processed
    finally:
        db.close()


def _ticket_retry_worker_loop() -> None:
    while not _ticket_retry_stop_event.is_set():
        try:
            _process_pending_ticket_payloads(batch_size=TICKET_RETRY_BATCH_SIZE)
        except Exception:
            logger.error("Ticket retry worker iteration failed", exc_info=True)
        _ticket_retry_stop_event.wait(TICKET_RETRY_INTERVAL)


def _start_ticket_retry_worker() -> None:
    global _ticket_retry_thread
    if _ticket_retry_thread and _ticket_retry_thread.is_alive():
        return
    _ticket_retry_stop_event.clear()
    _ticket_retry_thread = threading.Thread(
        target=_ticket_retry_worker_loop,
        name="ticket-payload-retry",
        daemon=True,
    )
    _ticket_retry_thread.start()


def _stop_ticket_retry_worker() -> None:
    global _ticket_retry_thread
    _ticket_retry_stop_event.set()
    thread = _ticket_retry_thread
    if thread and thread.is_alive():
        thread.join(timeout=5)
    _ticket_retry_thread = None

def require_permission(permission_name: str):
    """Dependency that checks the current user has a given permission."""

    def dependency(current_user: User = Depends(get_current_user)):
        for role in current_user.roles:
            if any(p.name == permission_name for p in role.permissions):
                return current_user
        raise HTTPException(status_code=403, detail="Not enough permissions")

    return dependency


@app.post("/users")
def create_user(
    user: UserCreate,
    current_user: User = Depends(require_permission("manage_users")),
):
    db = SessionLocal()
    try:
        if _retry_operation(
            lambda s: s.query(User).filter(User.username == user.username).first(), db
        ):
            raise HTTPException(status_code=400, detail="Username already exists")
        roles = []
        if user.role_ids:
            roles = _retry_operation(
                lambda s: s.query(Role).filter(Role.id.in_(user.role_ids)).all(),
                db,
            )
        new_user = User(
            username=user.username, hashed_password=get_password_hash(user.password)
        )
        new_user.roles = roles
        db.add(new_user)
        _retry_commit(new_user, db)
        return {"id": new_user.id}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()


@app.get("/users")
def list_users(current_user: User = Depends(require_permission("manage_users"))):
    db = SessionLocal()
    try:
        users = _retry_operation(
            lambda s: s.query(User).options(joinedload(User.roles)).all(),
            db,
        )
        return [{**_as_dict(u), "roles": [r.id for r in u.roles]} for u in users]
    finally:
        db.close()


@app.get("/users/{user_id}")
def get_user(
    user_id: int, current_user: User = Depends(require_permission("manage_users"))
):
    db = SessionLocal()
    try:
        u = _retry_operation(
            lambda s: s.query(User).options(joinedload(User.roles)).get(user_id),
            db,
        )
        if u is None:
            raise HTTPException(status_code=404, detail="Not found")
        return {**_as_dict(u), "roles": [r.id for r in u.roles]}
    finally:
        db.close()


@app.put("/users/{user_id}")
def update_user(
    user_id: int,
    user: UserUpdate,
    current_user: User = Depends(require_permission("manage_users")),
):
    db = SessionLocal()
    try:
        obj = _retry_operation(
            lambda s: s.query(User).options(joinedload(User.roles)).get(user_id),
            db,
        )
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        if user.username is not None:
            obj.username = user.username
        if user.password is not None:
            obj.hashed_password = get_password_hash(user.password)
        if user.role_ids is not None:
            obj.roles = _retry_operation(
                lambda s: s.query(Role).filter(Role.id.in_(user.role_ids)).all(),
                db,
            )
        _retry_commit(obj, db)
        return {**_as_dict(obj), "roles": [r.id for r in obj.roles]}
    finally:
        db.close()


@app.delete("/users/{user_id}")
def delete_user(
    user_id: int, current_user: User = Depends(require_permission("manage_users"))
):
    db = SessionLocal()
    try:
        obj = _retry_operation(lambda s: s.query(User).get(user_id), db)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        db.delete(obj)
        _retry_commit(obj, db)
        return {"status": "deleted"}
    finally:
        db.close()


@app.post("/roles")
def create_role(
    role: RoleCreate,
    current_user: User = Depends(require_permission("manage_roles")),
):
    db = SessionLocal()
    try:
        if _retry_operation(
            lambda s: s.query(Role).filter(Role.name == role.name).first(), db
        ):
            raise HTTPException(status_code=400, detail="Role already exists")
        perms = []
        if role.permission_ids:
            perms = _retry_operation(
                lambda s: s.query(Permission)
                .filter(Permission.id.in_(role.permission_ids))
                .all(),
                db,
            )
        new_role = Role(name=role.name, description=role.description)
        new_role.permissions = perms
        db.add(new_role)
        _retry_commit(new_role, db)
        return {"id": new_role.id}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()


@app.get("/roles")
def list_roles(current_user: User = Depends(require_permission("manage_roles"))):
    db = SessionLocal()
    try:
        roles = _retry_operation(
            lambda s: s.query(Role).options(joinedload(Role.permissions)).all(),
            db,
        )
        return [
            {**_as_dict(r), "permissions": [p.id for p in r.permissions]} for r in roles
        ]
    finally:
        db.close()


@app.get("/roles/{role_id}")
def get_role(
    role_id: int, current_user: User = Depends(require_permission("manage_roles"))
):
    db = SessionLocal()
    try:
        role = _retry_operation(
            lambda s: s.query(Role).options(joinedload(Role.permissions)).get(role_id),
            db,
        )
        if role is None:
            raise HTTPException(status_code=404, detail="Not found")
        return {**_as_dict(role), "permissions": [p.id for p in role.permissions]}
    finally:
        db.close()


@app.put("/roles/{role_id}")
def update_role(
    role_id: int,
    role: RoleUpdate,
    current_user: User = Depends(require_permission("manage_roles")),
):
    db = SessionLocal()
    try:
        obj = _retry_operation(
            lambda s: s.query(Role).options(joinedload(Role.permissions)).get(role_id),
            db,
        )
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        if role.name is not None:
            obj.name = role.name
        if role.description is not None:
            obj.description = role.description
        if role.permission_ids is not None:
            obj.permissions = _retry_operation(
                lambda s: s.query(Permission)
                .filter(Permission.id.in_(role.permission_ids))
                .all(),
                db,
            )
        _retry_commit(obj, db)
        return {**_as_dict(obj), "permissions": [p.id for p in obj.permissions]}
    finally:
        db.close()


@app.delete("/roles/{role_id}")
def delete_role(
    role_id: int, current_user: User = Depends(require_permission("manage_roles"))
):
    db = SessionLocal()
    try:
        obj = _retry_operation(lambda s: s.query(Role).get(role_id), db)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        db.delete(obj)
        _retry_commit(obj, db)
        return {"status": "deleted"}
    finally:
        db.close()


@app.post("/permissions")
def create_permission(
    perm: PermissionCreate,
    current_user: User = Depends(require_permission("manage_permissions")),
):
    db = SessionLocal()
    try:
        if _retry_operation(
            lambda s: s.query(Permission).filter(Permission.name == perm.name).first(),
            db,
        ):
            raise HTTPException(status_code=400, detail="Permission already exists")
        new_perm = Permission(name=perm.name, description=perm.description)
        db.add(new_perm)
        _retry_commit(new_perm, db)
        return {"id": new_perm.id}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()


@app.get("/permissions")
def list_permissions(
    current_user: User = Depends(require_permission("manage_permissions")),
):
    db = SessionLocal()
    try:
        perms = _retry_operation(lambda s: s.query(Permission).all(), db)
        return [_as_dict(p) for p in perms]
    finally:
        db.close()


@app.get("/permissions/{perm_id}")
def get_permission(
    perm_id: int, current_user: User = Depends(require_permission("manage_permissions"))
):
    db = SessionLocal()
    try:
        perm = _retry_operation(lambda s: s.query(Permission).get(perm_id), db)
        if perm is None:
            raise HTTPException(status_code=404, detail="Not found")
        return _as_dict(perm)
    finally:
        db.close()


@app.put("/permissions/{perm_id}")
def update_permission(
    perm_id: int,
    perm: PermissionUpdate,
    current_user: User = Depends(require_permission("manage_permissions")),
):
    db = SessionLocal()
    try:
        obj = _retry_operation(lambda s: s.query(Permission).get(perm_id), db)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        for k, v in perm.dict(exclude_unset=True).items():
            setattr(obj, k, v)
        _retry_commit(obj, db)
        return _as_dict(obj)
    finally:
        db.close()


@app.delete("/permissions/{perm_id}")
def delete_permission(
    perm_id: int, current_user: User = Depends(require_permission("manage_permissions"))
):
    db = SessionLocal()
    try:
        obj = _retry_operation(lambda s: s.query(Permission).get(perm_id), db)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        db.delete(obj)
        _retry_commit(obj, db)
        return {"status": "deleted"}
    finally:
        db.close()


def _retry_commit(obj, session):
    """
    Try to session.commit() on obj; if commit fails due to lost connection,
    rollback/close and retry once with a fresh session.
    """
    try:
        session.commit()
    except OperationalError:
        logger.warning("Lost DB connection during commit; retrying once", exc_info=True)
        try:
            session.rollback()
        except:
            pass
        try:
            session.close()
        except:
            pass

        new_sess = SessionLocal()
        try:
            # ``merge`` ensures the object is attached to the new session whether
            # it is a brand new entity or an existing one that was previously
            # persisted.  Using ``add`` could fail for existing objects because
            # SQLAlchemy would attempt to issue an INSERT for a row that already
            # exists.
            new_sess.merge(obj)
            new_sess.commit()
        finally:
            new_sess.close()


def _retry_operation(func, session):
    """Execute ``func(session)`` and retry once on ``OperationalError``.

    If the first attempt raises ``OperationalError``, the session is
    rolled back and closed, then the function is called again with a new
    fresh session.  Any return value from ``func`` is returned.
    """
    try:
        return func(session)
    except OperationalError:
        logger.warning(
            "Lost DB connection during operation; retrying once", exc_info=True
        )
        try:
            session.rollback()
        except Exception:
            pass
        try:
            session.close()
        except Exception:
            pass

        new_sess = SessionLocal()
        try:
            return func(new_sess)
        finally:
            new_sess.close()


def save_report_to_file(payload: dict, camera_id: int, spot_number: int, ts: str):
    """Write parking report payload to a JSON file under ``REPORTS_JSON_DIR``."""
    rand3 = random.randint(111111, 999999999999)
    filename = os.path.join(
        REPORTS_JSON_DIR, f"report_cam{camera_id}_spot{spot_number}_{rand3}.json"
    )
    try:
        with open(filename, "w") as f:
            json.dump(payload, f)
    except Exception:
        logger.error("Failed to write report JSON", exc_info=True)


def save_ignored_trigger(
    kind: str,
    camera_id: int,
    spot_number: int,
    ticket_id: int | str | None,
    payload: dict,
    snapshot_bytes: bytes | None,
    reason: str,
):
    """Save ignored entry/exit trigger information to disk."""
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    base_dir = IGNORED_ENTRY_DIR if kind == "entry" else IGNORED_EXIT_DIR
    rand4 = random.randint(111111, 999999999999)
    folder = os.path.join(
        base_dir,
        f"cam{camera_id}_spot{spot_number}_ticket{ticket_id if ticket_id is not None else 'none'}_{rand4}",
    )
    os.makedirs(folder, exist_ok=True)
    try:
        with open(os.path.join(folder, "request.json"), "w") as f:
            json.dump(payload, f)
    except Exception:
        logger.error("Failed to save ignored request JSON", exc_info=True)

    if snapshot_bytes:
        try:
            with open(os.path.join(folder, "snapshot.jpg"), "wb") as imgf:
                imgf.write(snapshot_bytes)
        except Exception:
            logger.error("Failed to save ignored snapshot", exc_info=True)

    try:
        with open(os.path.join(folder, "reason.txt"), "w") as lf:
            lf.write(reason)
    except Exception:
        logger.error("Failed to write ignored reason", exc_info=True)


def _process_plate_task(
    payload: dict,
    park_folder: str,
    ts: str,
    camera_id: int,
    pole_id: int,
    portal_id: int | None,
    spot_number: int,
    camera_ip: str,
    camera_user: str,
    camera_pass: str,
    parkonic_api_token: str,
    rtsp_path: str = "/",
):
    """Run plate processing synchronously in the worker thread."""
    process_plate_and_issue_ticket(
        payload,
        park_folder,
        ts,
        camera_id,
        pole_id,
        portal_id,
        spot_number,
        camera_ip,
        camera_user,
        camera_pass,
        parkonic_api_token,
        rtsp_path,
    )


def _exit_flow(
    payload: dict,
    ts: str,
    camera_id: int,
    portal_id: int | None,
    spot_number: int,
    camera_ip: str,
    cam_user: str,
    cam_pass: str,
    parkonic_api_token: str,
):
    """Simplified EXIT logic: immediately close any open ticket and
    send the ticket information to the external API using an annotated
    exit snapshot instead of a video clip. The annotated snapshot is
    saved locally and its path stored in ``exit_clip_path``."""

    db2 = SessionLocal()
    try:
        open_ticket = (
            db2.query(Ticket)
            .filter_by(camera_id=camera_id, spot_number=spot_number, exit_time=None)
            .order_by(Ticket.entry_time.desc())
            .first()
        )
        spot_obj = (
            db2.query(Spot)
            .filter_by(camera_id=camera_id, spot_number=spot_number)
            .first()
        )

        if open_ticket:
            entry_image_base64 = ""
            if open_ticket.entry_image_path and os.path.isfile(open_ticket.entry_image_path):
                try:
                    with open(open_ticket.entry_image_path, "rb") as f:
                        entry_image_base64 = base64.b64encode(f.read()).decode("utf-8")
                except Exception:
                    logger.error("Failed reading entry image for API", exc_info=True)

            exit_upload_name = ""
            exit_clip_local = None
            delivery_acknowledged = False
            delivery_response_received = False
            pending_record = None
            ticket_payload_kwargs: dict | None = None
            if payload.get("snapshot") and spot_obj:
                if open_ticket.confidence > 50:
                    try:
                        img_bytes = base64.b64decode(payload["snapshot"])
                        img = Image.open(io.BytesIO(img_bytes))
                        draw = ImageDraw.Draw(img)
                        left, top, right, bottom = spot_obj.bbox
                        draw.rectangle([left, top, right, bottom], outline="red", width=3)
                        rand_exit = random.randint(111111, 999999999999)
                        exit_clip_local = os.path.join(
                            SNAPSHOTS_DIR,
                            f"exit_{camera_id}_{spot_number}_{rand_exit}.jpg",
                        )
                        try:
                            img.save(exit_clip_local)
                        except Exception:
                            logger.error("Failed to save exit snapshot", exc_info=True)
                            exit_clip_local = None
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG")
                        buf.seek(0)
                        files = {"file": ("exit.jpg", buf.getvalue())}
                        try:
                            resp = http_session.post(
                                "http://10.11.5.103:18001/upload-video", files=files
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                exit_upload_name = data.get("file_name", "")
                        except Exception:
                            logger.error("Failed to upload exit snapshot", exc_info=True)
                    except Exception:
                        logger.error("Failed processing exit snapshot", exc_info=True)
                    exit_ts = datetime.strptime(
                        payload["time"], "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    ticket_payload_kwargs = {
                        "token": parkonic_api_token,
                        "access_point_id": portal_id,
                        "number": open_ticket.plate_number,
                        "code": open_ticket.plate_code,
                        "city": open_ticket.plate_city,
                        "status": "waiting",
                        "entry_time": open_ticket.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_time": exit_ts,
                        "entry_pic_base64": entry_image_base64,
                        "car_pic_base64": open_ticket.image_base64,
                        "exit_video_path": exit_upload_name,
                        "ticket_key_id": open_ticket.id,
                        "spot_number": open_ticket.spot_number,
                    }
                    try:
                        response = send_ticket_payload(**ticket_payload_kwargs)
                    except Exception as exc:
                        logger.error("Failed sending ticket payload", exc_info=True)
                        pending_record = _enqueue_pending_ticket_payload(
                            db2,
                            ticket_id=open_ticket.id,
                            payload_kwargs=ticket_payload_kwargs,
                            error_message=str(exc),
                        )
                    else:
                        delivery_response_received = True
                        if _is_ticket_delivery_acknowledged(response):
                            delivery_acknowledged = True
                        else:
                            logger.warning(
                                "Ticket payload for ticket id=%s was not acknowledged; response=%s",
                                open_ticket.id,
                                _stringify_response(response),
                            )

            should_finalize = False
            if ticket_payload_kwargs is None:
                should_finalize = True
            elif (
                delivery_acknowledged
                or pending_record is not None
                or delivery_response_received
            ):
                should_finalize = True

            if pending_record is not None:
                logger.info(
                    "Queued ticket payload id=%s for retry for ticket id=%s",
                    pending_record.id,
                    open_ticket.id,
                )

            if not should_finalize:
                logger.error(
                    "Ticket id=%s exit payload neither acknowledged nor queued; leaving ticket open",
                    open_ticket.id,
                )
                return JSONResponse(
                    status_code=502,
                    content={"message": "Failed to deliver exit event"},
                )

            open_ticket.exit_time = datetime.fromisoformat(payload["time"])
            open_ticket.exit_clip_path = exit_clip_local
            _retry_commit(open_ticket, db2)
            if spot_obj:
                spot_obj.status = 0
                _retry_commit(spot_obj, db2)

            logger.debug(
                "Closed ticket id=%d at %s camera %d spot %d",
                open_ticket.id,
                payload["time"],
                camera_id,
                spot_number,
            )

            return JSONResponse(status_code=200, content={"message": "Exit recorded"})

        if spot_obj:
            spot_obj.status = 0
            _retry_commit(spot_obj, db2)
        return JSONResponse(status_code=200, content={"message": "No open ticket to close"})
    except SQLAlchemyError as e:
        try:
            db2.rollback()
        except Exception:
            pass
        logger.error("Database error on EXIT", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database error on exit: {e}")
    finally:
        db2.close()


def _process_post_task(payload: dict, raw_body: bytes, ts: str):
    rand5 = random.randint(111111, 999999999999)
    # raw_fn = os.path.join(RAW_REQUEST_DIR, f"raw_request_{rand5}.json")
    # try:
    #     with open(raw_fn, "wb") as f:
    #         f.write(raw_body)
    # except Exception:
    #     logger.error("Failed to write raw request to disk", exc_info=True)
    required_fields = [
        "event",
        "device",
        "time",
        "report_type",
        "resolution_w",
        "resolution_y",
        "parking_area",
        "index_number",
        "occupancy",
        "duration",
        "coordinate_x1",
        "coordinate_y1",
        "coordinate_x2",
        "coordinate_y2",
        "coordinate_x3",
        "coordinate_y3",
        "coordinate_x4",
        "coordinate_y4",
        "vehicle_frame_x1",
        "vehicle_frame_y1",
        "vehicle_frame_x2",
        "vehicle_frame_y2",
        "snapshot",
    ]
    missing = [f for f in required_fields if payload.get(f) is None]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Missing fields: {', '.join(missing)}"
        )
    m = re.match(r"^([A-Za-z]+)(\d+)$", payload["parking_area"])
    if not m:
        raise HTTPException(
            status_code=400,
            detail="Invalid parking_area format (expected letters+digits, e.g. 'NAD95')",
        )
    location_code = m.group(1)
    api_code = m.group(2)
    spot_number = payload["index_number"]
    rtsp_path = "/"
    try:
        db = SessionLocal()
        stmt = text(
            """
            SELECT
              c.id      AS camera_id,
              c.pole_id AS pole_id,

              c.p_ip    AS camera_ip,
              c.portal_id AS portal_id,
              l.parkonic_api_token AS parkonic_api_token,
              l.camera_user        AS camera_user,
              l.camera_pass        AS camera_pass,
              l.parameters         AS location_params

            FROM cameras AS c
            JOIN poles     AS p ON c.pole_id   = p.id
            JOIN zones     AS z ON p.zone_id    = z.id
            JOIN locations AS l ON p.location_id = l.id
            WHERE l.code    = :loc_code
              AND c.api_code = :api_code
            LIMIT 1
            """
        )
        row = db.execute(
            stmt, {"loc_code": location_code, "api_code": api_code}
        ).fetchone()
        db.close()
        if row is None:
            raise HTTPException(
                status_code=400, detail="No camera found for that parking_area"
            )
        (
            camera_id,
            pole_id,
            camera_ip,
            portal_id,
            parkonic_api_token,
            cam_user,
            cam_pass,
            loc_params,
        ) = row
        rtsp_path = "/"
        if loc_params:
            try:
                if isinstance(loc_params, str):
                    loc_params = json.loads(loc_params)
                rtsp_path = (
                    loc_params.get("rtsp_path", "/")
                    if isinstance(loc_params, dict)
                    else "/"
                )
            except Exception:
                rtsp_path = "/"
    except OperationalError:
        logger.warning(
            "Lost DB connection during camera lookup; retrying once", exc_info=True
        )
        try:
            db.rollback()
        except Exception:
            pass
        try:
            db.close()
        except Exception:
            pass
        db = SessionLocal()
        row = db.execute(
            stmt, {"loc_code": location_code, "api_code": api_code}
        ).fetchone()
        db.close()
        if row is None:
            raise HTTPException(
                status_code=400, detail="No camera found for that parking_area"
            )
        (
            camera_id,
            pole_id,
            camera_ip,
            portal_id,
            parkonic_api_token,
            cam_user,
            cam_pass,
            loc_params,
        ) = row
        rtsp_path = "/"
        if loc_params:
            try:
                if isinstance(loc_params, str):
                    loc_params = json.loads(loc_params)
                rtsp_path = (
                    loc_params.get("rtsp_path", "/")
                    if isinstance(loc_params, dict)
                    else "/"
                )
            except Exception:
                rtsp_path = "/"
    if payload["occupancy"] == 0:
        return _exit_flow(
            payload,
            payload["time"],
            camera_id,
            portal_id,
            spot_number,
            camera_ip,
            cam_user,
            cam_pass,
            parkonic_api_token,
        )
    else:
        db2 = SessionLocal()
        try:
            existing_ticket = (
                db2.query(Ticket)
                .filter_by(camera_id=camera_id, spot_number=spot_number, exit_time=None)
                .order_by(Ticket.entry_time.desc())
                .first()
            )
            if existing_ticket:
                logger.debug(
                    "Spot %d on camera %d already occupied (ticket id=%d)",
                    spot_number,
                    camera_id,
                    existing_ticket.id,
                )
                try:
                    img_b = base64.b64decode(payload.get("snapshot", ""))
                except Exception:
                    img_b = None
                save_ignored_trigger(
                    "entry",
                    camera_id,
                    spot_number,
                    existing_ticket.id,
                    payload,
                    img_b,
                    "Spot already occupied",
                )
                return JSONResponse(
                    status_code=200, content={"message": "Spot already occupied"}
                )
            rand = random.randint(111111, 999999999999)
            save_report_to_file(payload, camera_id, spot_number, rand)
        except SQLAlchemyError as sa_err:
            try:
                db2.rollback()
            except Exception:
                pass
            logger.error("Database error during entry handling", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Database error on entry: {sa_err}"
            )
        finally:
            db2.close()
        park_folder = os.path.join(
            SNAPSHOTS_DIR, f"parking_cam{camera_id}_spot{spot_number}_{uuid.uuid4().hex}"
        )
        os.makedirs(park_folder, exist_ok=True)
        try:
            img_data = base64.b64decode(payload["snapshot"])
        except Exception as e:
            logger.error("Failed to decode snapshot", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Cannot decode snapshot: {e}")
        try:
            if not spot_has_car(
                img_data,
                camera_id=camera_id,
                spot_number=spot_number,
                save_folder=park_folder,
            ):
                logger.debug(
                    "ENTRY report ignored - no car detected. Camera=%d, Spot=%d",
                    camera_id,
                    spot_number,
                )
                save_ignored_trigger(
                    "entry",
                    camera_id,
                    spot_number,
                    None,
                    payload,
                    img_data,
                    "No car detected",
                )
                return JSONResponse(
                    status_code=200, content={"message": "No car detected"}
                )
        except Exception:
            logger.error("Error checking spot occupancy", exc_info=True)
        try:
            rand6 = random.randint(111111, 999999999999)
            snapshot_path = os.path.join(park_folder, f"snapshot_{rand6}.jpg")
            with open(snapshot_path, "wb") as imgf:
                imgf.write(img_data)
        except Exception as e:
            logger.error("Failed to save snapshot", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Cannot save snapshot: {e}")
        sess_update = SessionLocal()
        try:
            spot_obj = (
                sess_update.query(Spot)
                .filter_by(camera_id=camera_id, spot_number=spot_number)
                .first()
            )
            if spot_obj:
                spot_obj.status = 1
                sess_update.commit()
        except Exception:
            sess_update.rollback()
            logger.error("Failed to update spot status on entry", exc_info=True)
        finally:
            sess_update.close()
        _process_plate_task(
            payload,
            park_folder,
            payload["time"],
            camera_id,
            pole_id,
            portal_id,
            spot_number,
            camera_ip,
            cam_user,
            cam_pass,
            parkonic_api_token,
            rtsp_path,
        )
        return JSONResponse(
            status_code=200, content={"message": "Entry processed"}
        )




@app.post("/post")
async def receive_parking_data(request: Request):
    try:
        payload = await request.json()
    except Exception as e:
        logger.error("Failed to parse JSON payload", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"request_{ts}_{uuid.uuid4().hex}.json"
    file_path = os.path.join(RAW_REQUEST_DIR, file_name)
    try:
        with open(file_path, "w") as f:
            json.dump(payload, f)
    except Exception:
        logger.error("Failed to write raw request to disk", exc_info=True)

    _get_camera_id_from_payload(payload)
    try:
        _persist_camera_report(payload, require_snapshot=True)
    except HTTPException:
        raise
    except Exception:
        logger.error("Failed to persist camera report for /post", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to persist camera report")

    _enqueue_post_task(payload, b"", ts)

    return JSONResponse(
        status_code=200, content={"message": "Entry processed"}
    )


@app.post("/ocr-image")
async def ocr_image_endpoint(image: UploadFile = File(...)):
    img_bytes = await image.read()
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    arr = np.array(pil_img)
    results = ocr_processor.plate_model(arr)
    if not results or not results[0].boxes:
        raise HTTPException(status_code=400, detail="No plate detected")
    x1, y1, x2, y2 = results[0].boxes.xyxy[0].tolist()
    x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
    plate_crop = pil_img.crop((x1i, y1i, x2i, y2i))
    try:
        arr_bgr = cv2.cvtColor(np.array(plate_crop), cv2.COLOR_RGB2BGR)
        # arr_bgr = enhance_image_array(arr_bgr)
        plate_crop = Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))
    except Exception:
        logger.error("Plate enhancement failed", exc_info=True)
    buf = io.BytesIO()
    plate_crop.save(buf, format="JPEG")
    plate_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    ocr_payload = {
        "token": OCR_TOKEN,
        "base64": plate_b64,
    }
    try:
        ocr_resp = await send_request_with_retry_async(
            "https://parkonic.cloud/ParkonicJLT/anpr/engine/process",
            ocr_payload,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OCR request failed: {e}")
    intermediate = ocr_resp
    if isinstance(intermediate, str):
        try:
            intermediate = json.loads(intermediate)
        except Exception:
            pass
    if isinstance(intermediate, str):
        try:
            ocr_json = json.loads(intermediate)
        except Exception:
            ocr_json = None
    elif isinstance(intermediate, dict):
        ocr_json = intermediate
    else:
        ocr_json = None
    if ocr_json is None:
        raise HTTPException(status_code=500, detail="Invalid OCR response")
    return ocr_json


def _as_dict(model_obj):
    """Return a dict of column values for a SQLAlchemy model instance.

    Handles ``bytes``/``bytearray`` fields by converting them to a hex string so
    that FastAPI's ``jsonable_encoder`` won't attempt to decode them as UTF-8,
    which would raise ``UnicodeDecodeError``.
    """

    result = {}
    for c in model_obj.__table__.columns:
        value = getattr(model_obj, c.name)
        if isinstance(value, (bytes, bytearray, memoryview)):
            try:
                value = value.decode("utf-8")
            except Exception:
                value = value.hex()
        result[c.name] = value
    return result


@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    db = SessionLocal()
    try:
        user = _retry_operation(
            lambda s: (
                s.query(User)
                .options(joinedload(User.roles))
                .filter(User.username == form_data.username)
                .first()
            ),
            db,
        )
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=401, detail="Incorrect username or password"
            )
        role_names = [r.name for r in user.roles]
    finally:
        db.close()
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "roles": role_names},
        expires_delta=access_token_expires,
    )
    logger.debug(
        "Issued access token for user %s with roles %s", user.username, role_names
    )
    return {"access_token": access_token, "token_type": "bearer", "roles": role_names}


@app.post("/locations")
def create_location(
    loc: LocationCreate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        new_obj = Location(**loc.dict())
        db.add(new_obj)
        _retry_commit(new_obj, db)
        return {"id": new_obj.id}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()


@app.post("/zones")
def create_zone(
    zone: ZoneCreate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        new_obj = Zone(**zone.dict())
        db.add(new_obj)
        _retry_commit(new_obj, db)
        return {"id": new_obj.id}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()


@app.post("/poles")
def create_pole(
    pole: PoleCreate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        new_obj = Pole(**pole.dict())
        db.add(new_obj)
        _retry_commit(new_obj, db)
        return {"id": new_obj.id}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()


@app.post("/cameras")
def create_camera(
    cam: CameraCreate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        new_obj = Camera(**cam.dict())
        db.add(new_obj)
        _retry_commit(new_obj, db)
        return {"id": new_obj.id}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()


@app.get("/locations")
def list_locations(current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        objs = db.query(Location).order_by(desc(Location.created_at)).all()
        return [_as_dict(o) for o in objs]
    finally:
        db.close()


@app.get("/locations/{loc_id}")
def get_location(loc_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Location).get(loc_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        return _as_dict(obj)
    finally:
        db.close()


@app.put("/locations/{loc_id}")
def update_location(
    loc_id: int,
    loc: LocationUpdate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        obj = db.query(Location).get(loc_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        for k, v in loc.dict(exclude_unset=True).items():
            setattr(obj, k, v)
        _retry_commit(obj, db)
        return _as_dict(obj)
    finally:
        db.close()


@app.delete("/locations/{loc_id}")
def delete_location(loc_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Location).get(loc_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        db.delete(obj)
        _retry_commit(obj, db)
        return {"status": "deleted"}
    finally:
        db.close()


@app.get("/zones")
def list_zones(current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        objs = db.query(Zone).order_by(desc(Zone.id)).all()
        return [_as_dict(z) for z in objs]
    finally:
        db.close()


@app.get("/zones/{zone_id}")
def get_zone(zone_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Zone).get(zone_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        return _as_dict(obj)
    finally:
        db.close()


@app.put("/zones/{zone_id}")
def update_zone(
    zone_id: int,
    zone: ZoneUpdate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        obj = db.query(Zone).get(zone_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        for k, v in zone.dict(exclude_unset=True).items():
            setattr(obj, k, v)
        _retry_commit(obj, db)
        return _as_dict(obj)
    finally:
        db.close()


@app.delete("/zones/{zone_id}")
def delete_zone(zone_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Zone).get(zone_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        db.delete(obj)
        _retry_commit(obj, db)
        return {"status": "deleted"}
    finally:
        db.close()


@app.get("/poles")
def list_poles(current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        objs = db.query(Pole).order_by(desc(Pole.id)).all()
        return [_as_dict(p) for p in objs]
    finally:
        db.close()


@app.get("/poles/{pole_id}")
def get_pole(pole_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Pole).get(pole_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        return _as_dict(obj)
    finally:
        db.close()


@app.put("/poles/{pole_id}")
def update_pole(
    pole_id: int,
    pole: PoleUpdate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        obj = db.query(Pole).get(pole_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        for k, v in pole.dict(exclude_unset=True).items():
            setattr(obj, k, v)
        _retry_commit(obj, db)
        return _as_dict(obj)
    finally:
        db.close()


@app.delete("/poles/{pole_id}")
def delete_pole(pole_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Pole).get(pole_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        db.delete(obj)
        _retry_commit(obj, db)
        return {"status": "deleted"}
    finally:
        db.close()


@app.get("/cameras")
def list_cameras(current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        objs = db.query(Camera).order_by(desc(Camera.api_code)).all()
        return [_as_dict(c) for c in objs]
    finally:
        db.close()


def _camera_occupancy(location_id: int | None):
    db = SessionLocal()
    try:
        query = db.query(Camera)
        if location_id is not None:
            query = query.join(Pole).filter(Pole.location_id == location_id)
        cameras = query.order_by(desc(Camera.api_code)).all()

        cam_ids = [c.id for c in cameras]
        spots = (
            db.query(Spot)
            .filter(Spot.camera_id.in_(cam_ids))
            .order_by(asc(Spot.spot_number))
            .all()
        )
        spot_map: dict[int, dict[int, int]] = {}
        for s in spots:
            spot_map.setdefault(s.camera_id, {})[s.spot_number] = s.status

        result = []
        for cam in cameras:
            cam_spots = spot_map.get(cam.id, {})
            occupied = [num for num, st in cam_spots.items() if st == 1]
            data = _as_dict(cam)
            data["spot_count"] = len(cam_spots)
            data["occupied_count"] = len(occupied)
            data["occupied_spots"] = sorted(occupied)
            data["spots"] = cam_spots
            result.append(data)
        return result
    finally:
        db.close()


@app.get("/camera-occupancy")
def camera_occupancy_all(current_user: User = Depends(get_current_user)):
    """Return occupancy information for all cameras."""

    return _camera_occupancy(None)


@app.get("/camera-occupancy/{location_id}")
def camera_occupancy_by_location(
    location_id: int, current_user: User = Depends(get_current_user)
):
    """Return occupancy information for cameras in the given location."""

    return _camera_occupancy(location_id)


@app.get("/cameras/{cam_id}")
def get_camera(cam_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Camera).get(cam_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        return _as_dict(obj)
    finally:
        db.close()


@app.get("/cameras/{cam_id}/clip")
def get_camera_clip(
    cam_id: int,
    start: str,
    end: str,
    current_user: User = Depends(get_current_user),
):
    """Fetch a video clip from a camera between ``start`` and ``end``."""

    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format")

    if end_dt <= start_dt:
        raise HTTPException(status_code=400, detail="end must be after start")

    db = SessionLocal()
    try:
        row = db.execute(
            text(
                """
                SELECT c.p_ip, l.camera_user, l.camera_pass
                FROM cameras c
                JOIN poles p ON c.pole_id = p.id
                JOIN locations l ON p.location_id = l.id
                WHERE c.id = :cam_id
                LIMIT 1
                """
            ),
            {"cam_id": cam_id},
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Camera not found")

        cam_ip, user, pwd = row

    finally:
        db.close()

    clip_path = request_camera_clip(
        camera_ip=cam_ip,
        username=user or "",
        password=pwd or "",
        start_dt=start_dt,
        end_dt=end_dt,
        segment_name=start_dt.strftime("%Y%m%d%H%M%S"),
        unique_tag=str(cam_id),
    )

    if not clip_path or not os.path.isfile(clip_path) or not is_valid_mp4(clip_path):
        raise HTTPException(status_code=500, detail="Failed to fetch clip")

    return stream_file(clip_path)


@app.get("/cameras/{cam_id}/frame")
def get_camera_frame(
    cam_id: int,
    current_user: User = Depends(get_current_user),
):
    """Return a JPEG frame captured from the camera."""

    db = SessionLocal()
    try:
        row = db.execute(
            text(
                """
                SELECT c.p_ip, l.camera_user, l.camera_pass, l.parameters
                FROM cameras c
                JOIN poles p ON c.pole_id = p.id
                JOIN locations l ON p.location_id = l.id
                WHERE c.id = :cam_id
                LIMIT 1
                """
            ),
            {"cam_id": cam_id},
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Camera not found")

        cam_ip, user, pwd, params = row
        rtsp_path = "/"
        if params:
            try:
                if isinstance(params, str):
                    params = json.loads(params)
                rtsp_path = (
                    params.get("rtsp_path", "/") if isinstance(params, dict) else "/"
                )
            except Exception:
                rtsp_path = "/"
    finally:
        db.close()

    try:
        frame_bytes = fetch_camera_frame(cam_ip, user, pwd)
    except Exception:
        logger.error("Failed fetching camera frame", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch frame")

    return Response(content=frame_bytes, media_type="image/jpeg")


def _process_clip_request(
    req_id: int, cam_ip: str, user: str, pwd: str, start_dt: datetime, end_dt: datetime
):
    """Background task to fetch clip and update ClipRequest row."""
    clip_path = request_camera_clip(
        camera_ip=cam_ip,
        username=user or "",
        password=pwd or "",
        start_dt=start_dt,
        end_dt=end_dt,
        segment_name=start_dt.strftime("%Y%m%d%H%M%S"),
        unique_tag=str(req_id),
    )
    session = SessionLocal()
    try:
        req = session.query(ClipRequest).get(req_id)
        if req:
            if clip_path and os.path.isfile(clip_path) and is_valid_mp4(clip_path):
                req.status = "COMPLETED"
                req.clip_path = clip_path
            else:
                req.status = "FAILED"
            session.commit()
    except Exception:
        logger.error("Failed updating clip request %d", req_id, exc_info=True)
        session.rollback()
    finally:
        session.close()


async def _process_clip_request_async(
    req_id: int, cam_ip: str, user: str, pwd: str, start_dt: datetime, end_dt: datetime
):
    await run_in_executor(
        _process_clip_request,
        req_id,
        cam_ip,
        user,
        pwd,
        start_dt,
        end_dt,
    )


@app.post("/clip-requests")
async def create_clip_request(
    data: ClipRequestCreate,
    current_user: User = Depends(get_current_user),
):
    """Create a camera clip request."""

    if data.end <= data.start:
        raise HTTPException(status_code=400, detail="end must be after start")

    db = SessionLocal()
    try:
        row = db.execute(
            text(
                """
                SELECT c.p_ip, l.camera_user, l.camera_pass
                FROM cameras c
                JOIN poles p ON c.pole_id = p.id
                JOIN locations l ON p.location_id = l.id
                WHERE c.id = :cam_id
                LIMIT 1
                """
            ),
            {"cam_id": data.camera_id},
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Camera not found")

        cam_ip, user, pwd = row

        req = ClipRequest(
            camera_id=data.camera_id,
            start_time=data.start,
            end_time=data.end,
            status="PENDING",
        )
        db.add(req)
        _retry_commit(req, db)
        status = req.status
        await _process_clip_request_async(
            req.id,
            cam_ip,
            user or "",
            pwd or "",
            data.start,
            data.end,
        )
        return {"id": req.id, "status": status}
    finally:
        db.close()


@app.get("/clip-requests")
def list_clip_requests(current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        objs = db.query(ClipRequest).order_by(desc(ClipRequest.created_at)).all()
        return [_as_dict(o) for o in objs]
    finally:
        db.close()


@app.delete("/clip-requests/{req_id}")
def delete_clip_request(req_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(ClipRequest).get(req_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        path = obj.clip_path
        db.delete(obj)
        _retry_commit(obj, db)
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                logger.error("Failed deleting clip file %s", path, exc_info=True)
        return {"status": "deleted"}
    finally:
        db.close()


@app.put("/cameras/{cam_id}")
def update_camera(
    cam_id: int,
    cam: CameraUpdate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        obj = db.query(Camera).get(cam_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        for k, v in cam.dict(exclude_unset=True).items():
            setattr(obj, k, v)
        _retry_commit(obj, db)
        return _as_dict(obj)
    finally:
        db.close()


@app.delete("/cameras/{cam_id}")
def delete_camera(cam_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Camera).get(cam_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        db.delete(obj)
        _retry_commit(obj, db)
        return {"status": "deleted"}
    finally:
        db.close()


@app.post("/spots")
def create_spot(
    spot: SpotCreate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        if not db.query(Camera.id).filter(Camera.id == spot.camera_id).first():
            raise HTTPException(status_code=404, detail="Camera not found")
        new_obj = Spot(**spot.dict())
        db.add(new_obj)
        _retry_commit(new_obj, db)
        return {"id": new_obj.id}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()


@app.get("/spots")
def list_spots(current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        objs = db.query(Spot).order_by(desc(Spot.id)).all()
        return [_as_dict(o) for o in objs]
    finally:
        db.close()


@app.get("/spots/{spot_id}")
def get_spot(spot_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Spot).get(spot_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        return _as_dict(obj)
    finally:
        db.close()


@app.delete("/spots/{spot_id}")
def delete_camera(spot_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Spot).get(spot_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        db.delete(obj)
        _retry_commit(obj, db)
        return {"status": "deleted"}
    finally:
        db.close()


@app.get("/cameras/{cam_id}/spots")
def list_camera_spots(cam_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        if not db.query(Camera.id).filter(Camera.id == cam_id).first():
            raise HTTPException(status_code=404, detail="Camera not found")
        objs = (
            db.query(Spot)
            .filter(Spot.camera_id == cam_id)
            .order_by(asc(Spot.spot_number))
            .all()
        )
        return [_as_dict(o) for o in objs]
    finally:
        db.close()


@app.post("/crop-zones")
def create_crop_zone(
    zone: CropZoneCreate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        if not db.query(Camera.id).filter(Camera.id == zone.camera_id).first():
            raise HTTPException(status_code=404, detail="Camera not found")
        new_obj = CropZone(**zone.dict())
        db.add(new_obj)
        _retry_commit(new_obj, db)
        return {"id": new_obj.id}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()


@app.get("/crop-zones")
def list_crop_zones(current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        objs = db.query(CropZone).order_by(desc(CropZone.id)).all()
        return [_as_dict(o) for o in objs]
    finally:
        db.close()


@app.get("/crop-zones/{zone_id}")
def get_crop_zone(zone_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(CropZone).get(zone_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        return _as_dict(obj)
    finally:
        db.close()


@app.put("/crop-zones/{zone_id}")
def update_crop_zone(
    zone_id: int,
    zone: CropZoneUpdate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        obj = db.query(CropZone).get(zone_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        for k, v in zone.dict(exclude_unset=True).items():
            setattr(obj, k, v)
        _retry_commit(obj, db)
        return _as_dict(obj)
    finally:
        db.close()


@app.delete("/crop-zones/{zone_id}")
def delete_crop_zone(zone_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(CropZone).get(zone_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        db.delete(obj)
        _retry_commit(obj, db)
        return {"status": "deleted"}
    finally:
        db.close()


@app.get("/cameras/{cam_id}/crop-zones")
def list_camera_crop_zones(cam_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        if not db.query(Camera.id).filter(Camera.id == cam_id).first():
            raise HTTPException(status_code=404, detail="Camera not found")
        objs = (
            db.query(CropZone)
            .filter(CropZone.camera_id == cam_id)
            .order_by(asc(CropZone.id))
            .all()
        )
        return [_as_dict(o) for o in objs]
    finally:
        db.close()


@app.get("/tickets")
def list_tickets(
    page: int = 1,
    page_size: int = 50,
    search: str | None = None,
    camera_id: int | None = None,
    spot_number: int | None = None,
    plate_number: str | None = None,
    plate_code: str | None = None,
    plate_city: str | None = None,
    entry_start: datetime | None = None,
    entry_end: datetime | None = None,
    sort_by: str = "id",
    sort_order: str = "desc",
    current_user: User = Depends(get_current_user),
):
    """Return paginated list of tickets with optional search and sorting."""

    db = SessionLocal()
    try:
        query = db.query(Ticket)

        if search:
            pattern = f"%{search}%"
            query = query.filter(Ticket.plate_number.like(pattern))

        if camera_id is not None:
            query = query.filter(Ticket.camera_id == camera_id)
        if spot_number is not None:
            query = query.filter(Ticket.spot_number == spot_number)
        if plate_number is not None:
            query = query.filter(Ticket.plate_number == plate_number)
        if plate_code is not None:
            query = query.filter(Ticket.plate_code == plate_code)
        if plate_city is not None:
            query = query.filter(Ticket.plate_city == plate_city)
        if entry_start is not None:
            query = query.filter(Ticket.entry_time >= entry_start)
        if entry_end is not None:
            query = query.filter(Ticket.entry_time <= entry_end)

        sort_col = getattr(Ticket, sort_by, Ticket.id)
        order_fn = desc if sort_order.lower() == "desc" else asc
        query = query.order_by(order_fn(sort_col))

        total = query.count()
        results = query.offset((page - 1) * page_size).limit(page_size).all()

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "data": [_as_dict(t) for t in results],
        }
    finally:
        db.close()


@app.post("/tickets")
def create_ticket(
    ticket: TicketUpdate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        new_obj = Ticket(**ticket.dict(exclude_unset=True))
        db.add(new_obj)
        _retry_commit(new_obj, db)
        return {"id": new_obj.id}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Ticket).get(ticket_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        return _as_dict(obj)
    finally:
        db.close()


@app.get("/tickets/{ticket_id}/image")
def get_ticket_image(ticket_id: int):
    db = SessionLocal()
    try:
        ticket = db.query(Ticket).get(ticket_id)
        if (
            ticket is None
            or not ticket.entry_image_path
            or not os.path.isfile(ticket.entry_image_path)
        ):
            raise HTTPException(status_code=404, detail="Image not found")
        return stream_file(ticket.entry_image_path)
    finally:
        db.close()


@app.get("/tickets/{ticket_id}/exit-image")
def get_ticket_exit_image(ticket_id: int):
    db = SessionLocal()
    try:
        ticket = db.query(Ticket).get(ticket_id)
        if (
            ticket is None
            or not ticket.exit_clip_path
            or not os.path.isfile(ticket.exit_clip_path)
        ):
            raise HTTPException(status_code=404, detail="Image not found")
        return stream_file(ticket.exit_clip_path)
    finally:
        db.close()
@app.get("/tickets/{ticket_id}/video")
def get_ticket_video(ticket_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        ticket = db.query(Ticket).get(ticket_id)
        if (
            ticket is None
            or not ticket.exit_clip_path
            or not os.path.isfile(ticket.exit_clip_path)
        ):
            raise HTTPException(status_code=404, detail="Clip not found")
        return stream_file(ticket.exit_clip_path)
    finally:
        db.close()


@app.put("/tickets/{ticket_id}")
def update_ticket(
    ticket_id: int,
    ticket: TicketUpdate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        obj = db.query(Ticket).get(ticket_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        for k, v in ticket.dict(exclude_unset=True).items():
            setattr(obj, k, v)
        _retry_commit(obj, db)
        return _as_dict(obj)
    finally:
        db.close()


@app.delete("/tickets/{ticket_id}")
def delete_ticket(ticket_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Ticket).get(ticket_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        db.delete(obj)
        _retry_commit(obj, db)
        return {"status": "deleted"}
    finally:
        db.close()


@app.get("/reports")
def list_reports(current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        objs = db.query(Report).order_by(desc(Report.created_at)).all()
        return [_as_dict(r) for r in objs]
    finally:
        db.close()


@app.post("/reports")
def create_report(
    report: ReportUpdate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        new_obj = Report(**report.dict(exclude_unset=True))
        db.add(new_obj)
        _retry_commit(new_obj, db)
        return {"id": new_obj.id}
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        db.close()


@app.get("/reports/{report_id}")
def get_report(report_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Report).get(report_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        return _as_dict(obj)
    finally:
        db.close()


@app.put("/reports/{report_id}")
def update_report(
    report_id: int,
    report: ReportUpdate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        obj = db.query(Report).get(report_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        for k, v in report.dict(exclude_unset=True).items():
            setattr(obj, k, v)
        _retry_commit(obj, db)
        return _as_dict(obj)
    finally:
        db.close()


@app.delete("/reports/{report_id}")
def delete_report(report_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        obj = db.query(Report).get(report_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        db.delete(obj)
        _retry_commit(obj, db)
        return {"status": "deleted"}
    finally:
        db.close()


@app.get("/manual-reviews")
def list_manual_reviews(
    status: str = "PENDING",
    page: int = 1,
    page_size: int = 50,
    current_user: User = Depends(get_current_user),
):
    """Return paginated manual reviews filtered by status."""

    db = SessionLocal()
    try:
        query = db.query(ManualReview).filter_by(review_status=status)
        query = query.order_by(desc(ManualReview.created_at))

        total = query.count()
        reviews = query.offset((page - 1) * page_size).limit(page_size).all()

        data = [
            {
                "id": r.id,
                "camera_id": r.camera_id,
                "spot_number": r.spot_number,
                "event_time": r.event_time.isoformat(),
                "image_path": r.image_path,
                "clip_path": r.clip_path,
                "plate_status": r.plate_status,
            }
            for r in reviews
        ]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "data": data,
        }
    finally:
        db.close()


@app.get("/manual-reviews/{review_id}")
def get_manual_review(
    review_id: int,
    current_user: User = Depends(get_current_user),
):
    """Return a single manual review by id."""
    db = SessionLocal()
    try:
        obj = db.query(ManualReview).get(review_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="Not found")
        return _as_dict(obj)
    finally:
        db.close()


@app.get("/manual-reviews/{review_id}/image")
def get_review_image(
    review_id: int,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        review = db.query(ManualReview).get(review_id)
        if review is None or not os.path.isfile(review.image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        return stream_file(review.image_path)
    finally:
        db.close()


@app.get("/manual-reviews/{review_id}/video")
def get_review_video(
    review_id: int,
    # current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        review = db.query(ManualReview).get(review_id)
        if (
            review is None
            or not review.clip_path
            or not os.path.isfile(review.clip_path)
        ):
            raise HTTPException(status_code=404, detail="Clip not found")
        return stream_file(review.clip_path)
    finally:
        db.close()


@app.post("/manual-reviews/{review_id}/correct")
def correct_manual_review(
    review_id: int,
    correction: ManualCorrection,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        review = db.query(ManualReview).get(review_id)
        if review is None:
            raise HTTPException(status_code=404, detail="Review not found")
        if review.ticket_id is None:
            raise HTTPException(status_code=400, detail="No associated ticket")

        ticket = db.query(Ticket).get(review.ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")

        ticket.plate_number = correction.plate_number
        ticket.plate_code = correction.plate_code
        ticket.plate_city = correction.plate_city
        ticket.confidence = correction.confidence
        _retry_commit(ticket, db)

        review.review_status = "RESOLVED"
        review.plate_status = "READ"
        _retry_commit(review, db)

        try:
            from api_client import park_in_request

            park_token = None
            try:
                park_token = ticket.camera.pole.location.parkonic_api_token
            except Exception:
                park_token = None

            if ticket.image_base64:
                images_list = [ticket.image_base64]
            else:
                images_list = []
                folder = os.path.join(SNAPSHOTS_DIR, review.snapshot_folder)
                try:
                    for fname in os.listdir(folder):
                        if fname.startswith("annotated_") or fname.startswith(
                            "main_crop_"
                        ):
                            with open(os.path.join(folder, fname), "rb") as f:
                                images_list.append(
                                    base64.b64encode(f.read()).decode("utf-8")
                                )
                except Exception:
                    logger.error(
                        "Failed loading snapshot images for API", exc_info=True
                    )

                if not images_list:
                    with open(review.image_path, "rb") as f:
                        images_list = [base64.b64encode(f.read()).decode("utf-8")]

            portal_id = (
                db.query(Camera.portal_id)
                .filter(Camera.id == review.camera_id)
                .scalar()
            )

            ticket_resp = park_in_request(
                token=park_token or "",
                parkin_time=str(ticket.entry_time),
                plate_code=correction.plate_code,
                plate_number=correction.plate_number,
                emirates=correction.plate_city,
                conf=str(correction.confidence),
                spot_number=ticket.spot_number,
                pole_id=portal_id,
                images=images_list,
            )
            try:
                ticket_resp = json.loads(ticket_resp)
            except Exception:
                logger.error("Failed to parse ticket_resp", exc_info=True)
                ticket_resp = {}
            logger.info("[REVIEW] Park-in response: %s", ticket_resp)
            trip_id = (
                ticket_resp.get("trip_id") if isinstance(ticket_resp, dict) else None
            )
            logger.info("[REVIEW] Extracted trip_id: %s", trip_id)
            ticket2 = db.query(Ticket).get(review.ticket_id)
            ticket2.parkonic_trip_id = trip_id
            _retry_commit(ticket2, db)
            if ticket2 is None:
                raise HTTPException(status_code=404, detail="Ticket not found")
            if ticket2.exit_time is not None:
                try:
                    from api_client import park_out_request2

                    park_out_request2(
                        token=park_token or "",
                        parkout_time=str(ticket2.exit_time),
                        plate_code=ticket2.plate_code or "",
                        plate_number=ticket2.plate_number,
                        emirates=ticket2.plate_city or "",
                        conf=str(ticket2.confidence or 0),
                        spot_number=ticket2.spot_number,
                        pole_id=portal_id,
                    )
                except Exception:
                    logger.error("park_out_request failed", exc_info=True)
        except Exception:
            logger.error("park_in_request failed", exc_info=True)

        return {"status": "updated"}
    finally:
        db.close()


class ExternalManualCorrection(BaseModel):
    review_id: int
    plate_number: str
    plate_code: str
    plate_city: str
    image_base64: str


@app.post("/external-corrections")
def external_manual_correction(correction: ExternalManualCorrection):
    """Receive manual review corrections from an external client."""

    db = SessionLocal()
    try:
        review = db.query(ManualReview).get(correction.review_id)
        if review is None:
            raise HTTPException(status_code=404, detail="Review not found")
        if review.ticket_id is None:
            raise HTTPException(status_code=400, detail="No associated ticket")

        ticket = db.query(Ticket).get(review.ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="Ticket not found")

        portal_id = ticket.camera.portal_id if ticket.camera else None
        existing_closed = (
            db.query(Ticket)
            .join(Camera)
            .filter(
                Ticket.id != ticket.id,
                Ticket.spot_number == ticket.spot_number,
                Camera.portal_id == portal_id,
                Ticket.exit_time.isnot(None),
                (Ticket.plate_number == correction.plate_number),
                (Ticket.plate_code == correction.plate_code),
                (Ticket.plate_city == correction.plate_city),
            )
            .order_by(Ticket.entry_time.desc())
            .first()
        )

        if existing_closed:
            existing_closed.exit_time = None
            existing_closed.exit_clip_path = None
            existing_closed.plate_number = correction.plate_number
            existing_closed.plate_code = correction.plate_code
            existing_closed.plate_city = correction.plate_city
            existing_closed.image_base64 = correction.image_base64
            _retry_commit(existing_closed, db)

            review.ticket_id = existing_closed.id
            review.review_status = "RESOLVED"
            review.plate_status = "READ"
            _retry_commit(review, db)

            db.delete(ticket)
            _retry_commit(ticket, db)
            return {"status": "updated"}

        ticket.plate_number = correction.plate_number
        ticket.plate_code = correction.plate_code
        ticket.plate_city = correction.plate_city
        ticket.image_base64 = correction.image_base64
        _retry_commit(ticket, db)

        review.review_status = "RESOLVED"
        review.plate_status = "READ"
        _retry_commit(review, db)

        # Update ticket information locally without sending data to the portal

        return {"status": "updated"}
    finally:
        db.close()


@app.post("/manual-reviews/{review_id}/dismiss")
def dismiss_manual_review(
    review_id: int,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        review = db.query(ManualReview).get(review_id)
        if review is None:
            raise HTTPException(status_code=404, detail="Review not found")

        if review.ticket_id:
            ticket = db.query(Ticket).get(review.ticket_id)
            if ticket and ticket.exit_time is None:
                ticket.exit_time = ticket.entry_time
                _retry_commit(ticket, db)

        review.review_status = "RESOLVED"
        _retry_commit(review, db)
        return {"status": "dismissed"}
    finally:
        db.close()


@app.get("/manual-reviews/{review_id}/snapshots")
def list_review_snapshots(
    review_id: int,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        review = db.query(ManualReview).get(review_id)
        if review is None:
            raise HTTPException(status_code=404, detail="Review not found")
        folder = os.path.join(SNAPSHOTS_DIR, review.snapshot_folder)
        if not os.path.isdir(folder):
            raise HTTPException(status_code=404, detail="Snapshot folder not found")
        files = [
            f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
        ]
        files.sort(
            key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True
        )
        return {"files": files}
    finally:
        db.close()


@app.get("/manual-reviews/{review_id}/snapshots/{filename}")
def get_review_snapshot(
    review_id: int,
    filename: str,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        review = db.query(ManualReview).get(review_id)
        if review is None:
            raise HTTPException(status_code=404, detail="Review not found")
        folder = Path(SNAPSHOTS_DIR) / review.snapshot_folder
        path = (folder / filename).resolve()
        snapshots_root = Path(SNAPSHOTS_DIR).resolve()
        try:
            path.relative_to(snapshots_root)
        except ValueError:
            raise HTTPException(status_code=404, detail="File not found")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return stream_file(str(path))
    finally:
        db.close()


@app.get("/location-stats")
def location_stats(current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        locations = db.query(Location).all()
        data: list[dict] = []
        for loc in locations:
            loc_info = {
                "id": loc.id,
                "name": loc.name,
                "code": loc.code,
                "zone_count": db.query(func.count(Zone.id))
                .filter(Zone.location_id == loc.id)
                .scalar()
                or 0,
                "zones": [],
            }
            zones = db.query(Zone).filter(Zone.location_id == loc.id).all()
            for zone in zones:
                zone_info = {
                    "id": zone.id,
                    "code": zone.code,
                    "pole_count": db.query(func.count(Pole.id))
                    .filter(Pole.zone_id == zone.id)
                    .scalar()
                    or 0,
                    "poles": [],
                }
                poles = db.query(Pole).filter(Pole.zone_id == zone.id).all()
                for pole in poles:
                    camera_count = (
                        db.query(func.count(Camera.id))
                        .filter(Camera.pole_id == pole.id)
                        .scalar()
                        or 0
                    )
                    ticket_count = (
                        db.query(func.count(Ticket.id))
                        .join(Camera, Ticket.camera_id == Camera.id)
                        .filter(Camera.pole_id == pole.id)
                        .scalar()
                        or 0
                    )
                    review_count = (
                        db.query(func.count(ManualReview.id))
                        .join(Camera, ManualReview.camera_id == Camera.id)
                        .filter(Camera.pole_id == pole.id)
                        .scalar()
                        or 0
                    )
                    pole_info = {
                        "id": pole.id,
                        "code": pole.code,
                        "camera_count": camera_count,
                        "ticket_count": ticket_count,
                        "manual_review_count": review_count,
                    }
                    zone_info["poles"].append(pole_info)
                loc_info["zones"].append(zone_info)
            data.append(loc_info)
        return {"data": data}
    finally:
        db.close()


@app.post("/manual-reviews/send-videos")
def send_pending_review_videos(
    server_base: str = "http://10.11.5.100:18007",
):
    """Send videos for all pending manual reviews to the external API."""

    db = SessionLocal()
    dispatched = 0
    try:
        reviews = (
            db.query(ManualReview)
            .filter(
                ManualReview.review_status == "PENDING",
                ManualReview.clip_path.isnot(None),
            )
            .all()
        )

        for review in reviews:
            if not os.path.isfile(review.clip_path or ""):
                continue

            crop_objs = (
                db.query(CropZone).filter(CropZone.camera_id == review.camera_id).all()
            )
            crop_list = [c.points for c in crop_objs]
            spot_obj = (
                db.query(Spot)
                .filter_by(camera_id=review.camera_id, spot_number=review.spot_number)
                .first()
            )
            spot_details = {}
            if spot_obj:
                spot_details = {
                    "p1_x": spot_obj.p1_x,
                    "p1_y": spot_obj.p1_y,
                    "p2_x": spot_obj.p2_x,
                    "p2_y": spot_obj.p2_y,
                    "p3_x": spot_obj.p3_x,
                    "p3_y": spot_obj.p3_y,
                    "p4_x": spot_obj.p4_x,
                    "p4_y": spot_obj.p4_y,
                }

            try:
                from api_client import send_review_video

                send_review_video(
                    review_id=review.id,
                    camera_id=review.camera_id,
                    spot_number=review.spot_number,
                    crop_zones=crop_list,
                    spot_details=spot_details,
                    video_path=review.clip_path,
                    server_base=server_base,
                )
                dispatched += 1
            except Exception:
                logger.error("send_review_video failed", exc_info=True)

        return {"dispatched": dispatched}
    finally:
        db.close()


@app.post("/manual-reviews/send-one-video")
def send_one_review_video_per_camera(
    server_base: str = "http://10.11.5.100:18007",
):
    """Send the first pending manual review for each camera."""

    db = SessionLocal()
    dispatched = 0
    try:
        camera_ids = db.query(Camera.id).all()

        for (cam_id,) in camera_ids:
            reviews = (
                db.query(ManualReview)
                .filter(
                    ManualReview.camera_id == cam_id,
                    ManualReview.review_status == "PENDING",
                    ManualReview.clip_path.isnot(None),
                )
                .order_by(ManualReview.id)
                .all()
            )

            review = None
            for r in reviews:
                if os.path.isfile(r.clip_path or ""):
                    review = r
                    break
            if review is None:
                continue

            crop_objs = (
                db.query(CropZone).filter(CropZone.camera_id == review.camera_id).all()
            )
            crop_list = [c.points for c in crop_objs]
            spot_obj = (
                db.query(Spot)
                .filter_by(camera_id=review.camera_id, spot_number=review.spot_number)
                .first()
            )
            spot_details = {}
            if spot_obj:
                spot_details = {
                    "p1_x": spot_obj.p1_x,
                    "p1_y": spot_obj.p1_y,
                    "p2_x": spot_obj.p2_x,
                    "p2_y": spot_obj.p2_y,
                    "p3_x": spot_obj.p3_x,
                    "p3_y": spot_obj.p3_y,
                    "p4_x": spot_obj.p4_x,
                    "p4_y": spot_obj.p4_y,
                }

            try:
                from api_client import send_review_video

                send_review_video(
                    review_id=review.id,
                    camera_id=review.camera_id,
                    spot_number=review.spot_number,
                    crop_zones=crop_list,
                    spot_details=spot_details,
                    video_path=review.clip_path,
                    server_base=server_base,
                )
                dispatched += 1
            except Exception:
                logger.error("send_review_video failed", exc_info=True)

        return {"dispatched": dispatched}
    finally:
        db.close()


@app.post("/manual-reviews/delete-missing-videos")
def delete_reviews_missing_videos():
    """Delete manual reviews lacking a video clip or whose file is missing."""

    db = SessionLocal()
    deleted = 0
    try:
        reviews = db.query(ManualReview).all()
        for r in reviews:
            if not r.clip_path or not os.path.isfile(r.clip_path):
                db.delete(r)
                _retry_commit(r, db)
                deleted += 1
        return {"deleted": deleted}
    finally:
        db.close()


@app.post("/echo")
async def echo(request: Request):
    """Echo back the request payload."""

    try:
        return await request.json()
    except Exception:
        data = await request.body()
        return {"data": data.decode()}
