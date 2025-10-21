# ocr_processor.py

import os
import shutil
import base64
import json
import io
import time
from datetime import datetime, timedelta
import random
import numpy as np
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
from PIL import Image, ImageDraw

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

from camera_clip import request_camera_clip
from network import send_request_with_retry

from config import OCR_TOKEN, VEHICLE_CLASSES, YOLO_MODEL_PATH
from image_enhancer import enhance_image_array

from models import PlateLog, Ticket, ManualReview, Spot, CropZone
from db import SessionLocal
from logger import logger
from utils import is_same_car, is_same_image
from logic import crop_and_save_car,same_entired_car

# Directories
PLATES_READ_DIR   = "plates/read"
PLATES_UNREAD_DIR = "plates/unread"
SPOT_LAST_DIR     = "spot_last"

os.makedirs(PLATES_READ_DIR,   exist_ok=True)
os.makedirs(PLATES_UNREAD_DIR, exist_ok=True)
os.makedirs(SPOT_LAST_DIR,      exist_ok=True)

# Load YOLO model lazily to avoid optional dependency failures
plate_model = None
_plate_model_attempted = False


def _load_plate_model():
    global plate_model, _plate_model_attempted
    if plate_model is not None:
        return plate_model
    if _plate_model_attempted:
        return None
    _plate_model_attempted = True
    if YOLO is None:  # pragma: no cover - optional dependency
        logger.warning("Ultralytics YOLO not available; OCR plate detection disabled")
        return None
    try:  # pragma: no cover - heavy optional dependency
        model = YOLO(YOLO_MODEL_PATH)
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        model.to(device)
        plate_model = model
    except Exception as exc:
        logger.warning("Failed to load plate detection model: %s", exc)
        plate_model = None
    return plate_model


def _get_plate_model():
    model = plate_model
    if model is None:
        model = _load_plate_model()
    return model

def spot_has_car(
    image: Image.Image | bytes,
    camera_id: int,
    spot_number: int,
    save_folder: str | None = None,
) -> bool:
    """Return ``True`` if a detected car still occupies the parking spot.

    The parking spot is stored as four corner points defining a polygon. The
    entire frame is processed by YOLO and each detected bounding box is
    considered part of the spot when its center lies inside that polygon and at
    least 65% of the box area overlaps with the polygon. When ``save_folder`` is
    provided an annotated debug image is saved showing the spot, detected cars
    and the intersection region for the matching car.
    """
    if isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    else:
        img = image

    db = SessionLocal()
    try:
        spot = (
            db.query(Spot)
            .filter_by(camera_id=camera_id, spot_number=spot_number)
            .first()
        )
    finally:
        db.close()

    if spot is None:
        return False

    spot_poly = [
        (float(spot.p1_x), float(spot.p1_y)),
        (float(spot.p2_x), float(spot.p2_y)),
        (float(spot.p3_x), float(spot.p3_y)),
        (float(spot.p4_x), float(spot.p4_y)),
    ]

    def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y):
                x_int = (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
                if x < x_int:
                    inside = not inside
            j = i
        return inside

    def clip_edge(poly, inside, intersect):
        if not poly:
            return []
        output = []
        prev = poly[-1]
        prev_in = inside(prev)
        for curr in poly:
            curr_in = inside(curr)
            if curr_in:
                if not prev_in:
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif prev_in:
                output.append(intersect(prev, curr))
            prev, prev_in = curr, curr_in
        return output

    def clip_polygon_rect(poly, rect):
        x_min, y_min, x_max, y_max = rect

        def inter_vert(p1, p2, x):
            if p1[0] == p2[0]:
                return (x, p1[1])
            t = (x - p1[0]) / (p2[0] - p1[0])
            return (x, p1[1] + t * (p2[1] - p1[1]))

        def inter_horiz(p1, p2, y):
            if p1[1] == p2[1]:
                return (p1[0], y)
            t = (y - p1[1]) / (p2[1] - p1[1])
            return (p1[0] + t * (p2[0] - p1[0]), y)

        poly = clip_edge(poly, lambda p: p[0] >= x_min, lambda s, e: inter_vert(s, e, x_min))
        poly = clip_edge(poly, lambda p: p[0] <= x_max, lambda s, e: inter_vert(s, e, x_max))
        poly = clip_edge(poly, lambda p: p[1] >= y_min, lambda s, e: inter_horiz(s, e, y_min))
        poly = clip_edge(poly, lambda p: p[1] <= y_max, lambda s, e: inter_horiz(s, e, y_max))
        return poly

    def polygon_area(poly) -> float:
        if len(poly) < 3:
            return 0.0
        area = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    model = _get_plate_model()
    if model is None:
        return False

    arr = np.array(img)
    orig_w, orig_h = img.size
    # Use full-HD resolution for YOLO processing to improve detection
    # accuracy when determining if a parking spot is occupied.
    model_w, model_h = 1920, 1080
    scale_x = 1.0
    scale_y = 1.0

    input_arr = arr
    if orig_w > model_w or orig_h > model_h:
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h
        resized = img.resize((model_w, model_h))
        input_arr = np.array(resized)

    results = model(input_arr, classes=VEHICLE_CLASSES)
    if not results or not results[0].boxes:
        return False

    hit_box: tuple[float, float, float, float] | None = None
    inter_box: tuple[float, float, float, float] | None = None

    for x1, y1, x2, y2 in results[0].boxes.xyxy.tolist():
        x1 *= scale_x
        y1 *= scale_y
        x2 *= scale_x
        y2 *= scale_y
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if not point_in_polygon(cx, cy, spot_poly):
            continue

        inter_poly = clip_polygon_rect(spot_poly, (x1, y1, x2, y2))
        inter_area = polygon_area(inter_poly)
        car_area = max((x2 - x1) * (y2 - y1), 1e-6)
        if inter_area / car_area >= 0.65:
            hit_box = (x1, y1, x2, y2)
            if inter_poly:
                ix = [p[0] for p in inter_poly]
                iy = [p[1] for p in inter_poly]
                inter_box = (min(ix), min(iy), max(ix), max(iy))
            break

    if save_folder and cv2 is not None:
        try:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            points = np.array(spot_poly, np.int32)
            # Reshape to match cv2.polylines requirement
            points = points.reshape((-1, 1, 2))

            # Draw the polygon outline
            cv2.polylines(img_bgr, [points], isClosed=True, color=(255, 255, 0), thickness=2)
            # cv2.rectangle(
            #     img_bgr,
            #     (int(left), int(top)),
            #     (int(right), int(bottom)),
            #     (255, 255, 0),
            #     2,
            # )
            if results and results[0].boxes:
                for bx in results[0].boxes.xyxy.tolist():
                    bx = [bx[0] * scale_x, bx[1] * scale_y, bx[2] * scale_x, bx[3] * scale_y]
                    color = (0, 0, 255)
                    if hit_box and tuple(bx) == tuple(hit_box):
                        color = (0, 255, 0)
                    cv2.rectangle(
                        img_bgr,
                        (int(bx[0]), int(bx[1])),
                        (int(bx[2]), int(bx[3])),
                        color,
                        2,
                    )
            if inter_box:
                cv2.rectangle(
                    img_bgr,
                    (int(inter_box[0]), int(inter_box[1])),
                    (int(inter_box[2]), int(inter_box[3])),
                    (255, 0, 255),
                    2,
                )
            os.makedirs(save_folder, exist_ok=True)
            debug_path = os.path.join(
                save_folder,
                f"spot_check_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg",
            )
            cv2.imwrite(debug_path, img_bgr)
        except Exception:
            logger.error("Failed to save spot_has_car debug image", exc_info=True)

    return hit_box is not None


def exit_decision_from_frame(
    image: Image.Image | bytes,
    camera_id: int,
    spot_number: int,
    ticket_id: int,
    save_folder: str | None = None,
) -> tuple[bool, bool]:
    """Check a frame for a car and whether it matches the given ticket.

    Returns a tuple ``(car_in_spot, ocr_match)`` where ``car_in_spot`` indicates
    if any detected vehicle sufficiently overlaps with the parking spot and
    ``ocr_match`` shows if that vehicle matches the ticket using
    :func:`same_entired_car`.
    """

    if isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    else:
        img = image

    db = SessionLocal()
    try:
        spot = (
            db.query(Spot)
            .filter_by(camera_id=camera_id, spot_number=spot_number)
            .first()
        )
    finally:
        db.close()

    if spot is None:
        return False, False

    spot_poly = [
        (float(spot.p1_x), float(spot.p1_y)),
        (float(spot.p2_x), float(spot.p2_y)),
        (float(spot.p3_x), float(spot.p3_y)),
        (float(spot.p4_x), float(spot.p4_y)),
    ]

    if cv2 is None:
        return False, False

    model = _get_plate_model()
    if model is None:
        return False, False

    def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y):
                x_int = (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
                if x < x_int:
                    inside = not inside
            j = i
        return inside

    def clip_edge(poly, inside, intersect):
        if not poly:
            return []
        output = []
        prev = poly[-1]
        prev_in = inside(prev)
        for curr in poly:
            curr_in = inside(curr)
            if curr_in:
                if not prev_in:
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif prev_in:
                output.append(intersect(prev, curr))
            prev, prev_in = curr, curr_in
        return output

    def clip_polygon_rect(poly, rect):
        x_min, y_min, x_max, y_max = rect

        def inter_vert(p1, p2, x):
            if p1[0] == p2[0]:
                return (x, p1[1])
            t = (x - p1[0]) / (p2[0] - p1[0])
            return (x, p1[1] + t * (p2[1] - p1[1]))

        def inter_horiz(p1, p2, y):
            if p1[1] == p2[1]:
                return (p1[0], y)
            t = (y - p1[1]) / (p2[1] - p1[1])
            return (p1[0] + t * (p2[0] - p1[0]), y)

        poly = clip_edge(poly, lambda p: p[0] >= x_min, lambda s, e: inter_vert(s, e, x_min))
        poly = clip_edge(poly, lambda p: p[0] <= x_max, lambda s, e: inter_vert(s, e, x_max))
        poly = clip_edge(poly, lambda p: p[1] >= y_min, lambda s, e: inter_horiz(s, e, y_min))
        poly = clip_edge(poly, lambda p: p[1] <= y_max, lambda s, e: inter_horiz(s, e, y_max))
        return poly

    def polygon_area(poly) -> float:
        if len(poly) < 3:
            return 0.0
        area = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    arr = np.array(img)
    orig_w, orig_h = img.size
    model_w, model_h = 1920, 1080
    scale_x = 1.0
    scale_y = 1.0

    input_arr = arr
    if orig_w > model_w or orig_h > model_h:
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h
        resized = img.resize((model_w, model_h))
        input_arr = np.array(resized)

    results = model(input_arr, classes=VEHICLE_CLASSES)
    if not results or not results[0].boxes:
        return False, False

    car_in_spot = False
    ocr_match = False
    hit_box: tuple[float, float, float, float] | None = None
    inter_box: tuple[float, float, float, float] | None = None

    for x1, y1, x2, y2 in results[0].boxes.xyxy.tolist():
        x1 *= scale_x
        y1 *= scale_y
        x2 *= scale_x
        y2 *= scale_y
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if not point_in_polygon(cx, cy, spot_poly):
            continue

        inter_poly = clip_polygon_rect(spot_poly, (x1, y1, x2, y2))
        inter_area = polygon_area(inter_poly)
        car_area = max((x2 - x1) * (y2 - y1), 1e-6)
        if inter_area / car_area >= 0.65:
            car_in_spot = True
            img2 = Image.open(io.BytesIO(image)).convert("RGB")
            frame = (
                cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
                if cv2 is not None
                else np.array(img2)
            )
            crop_path = crop_and_save_car(frame, int(x1), int(y1), int(x2), int(y2))
            ocr_match = same_entired_car(ticket_id, crop_path)
            if inter_poly and ocr_match:
                hit_box = (x1, y1, x2, y2)
                ix = [p[0] for p in inter_poly]
                iy = [p[1] for p in inter_poly]
                inter_box = (min(ix), min(iy), max(ix), max(iy))
            break

    if save_folder and cv2 is not None:
        try:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            points = np.array(spot_poly, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(img_bgr, [points], isClosed=True, color=(255, 255, 0), thickness=2)
            if results and results[0].boxes:
                for bx in results[0].boxes.xyxy.tolist():
                    bx = [bx[0] * scale_x, bx[1] * scale_y, bx[2] * scale_x, bx[3] * scale_y]
                    color = (0, 0, 255)
                    if hit_box and tuple(bx) == tuple(hit_box):
                        color = (0, 255, 0)
                    cv2.rectangle(img_bgr, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), color, 2)
            if inter_box:
                cv2.rectangle(
                    img_bgr,
                    (int(inter_box[0]), int(inter_box[1])),
                    (int(inter_box[2]), int(inter_box[3])),
                    (255, 0, 255),
                    2,
                )
            os.makedirs(save_folder, exist_ok=True)
            debug_path = os.path.join(
                save_folder,
                f"spot_check_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg",
            )
            cv2.imwrite(debug_path, img_bgr)
        except Exception:
            logger.error("Failed to save spot_has_car debug image", exc_info=True)

    return car_in_spot, ocr_match

def  spot_has_car_with_id(
    image: Image.Image | bytes,
    camera_id: int,
    spot_number: int,
    ticket_id:int,
    save_folder: str | None = None,
) -> bool:
    """Return ``True`` if a detected car still occupies the parking spot.

    The parking spot is stored as four corner points defining a polygon. The
    entire frame is processed by YOLO and each detected bounding box is
    considered part of the spot when its center lies inside that polygon and at
    least 65% of the box area overlaps with the polygon. When ``save_folder`` is
    provided an annotated debug image is saved showing the spot, detected cars
    and the intersection region for the matching car.
    """
    
    if isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    else:
        img = image

    db = SessionLocal()
    try:
        spot = (
            db.query(Spot)
            .filter_by(camera_id=camera_id, spot_number=spot_number)
            .first()
        )
    finally:
        db.close()

    if spot is None:
        return False

    spot_poly = [
        (float(spot.p1_x), float(spot.p1_y)),
        (float(spot.p2_x), float(spot.p2_y)),
        (float(spot.p3_x), float(spot.p3_y)),
        (float(spot.p4_x), float(spot.p4_y)),
    ]

    model = _get_plate_model()
    if model is None:
        return False
    


    def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y):
                x_int = (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
                if x < x_int:
                    inside = not inside
            j = i
        return inside

    def clip_edge(poly, inside, intersect):
        if not poly:
            return []
        output = []
        prev = poly[-1]
        prev_in = inside(prev)
        for curr in poly:
            curr_in = inside(curr)
            if curr_in:
                if not prev_in:
                    output.append(intersect(prev, curr))
                output.append(curr)
            elif prev_in:
                output.append(intersect(prev, curr))
            prev, prev_in = curr, curr_in
        return output

    def clip_polygon_rect(poly, rect):
        x_min, y_min, x_max, y_max = rect

        def inter_vert(p1, p2, x):
            if p1[0] == p2[0]:
                return (x, p1[1])
            t = (x - p1[0]) / (p2[0] - p1[0])
            return (x, p1[1] + t * (p2[1] - p1[1]))

        def inter_horiz(p1, p2, y):
            if p1[1] == p2[1]:
                return (p1[0], y)
            t = (y - p1[1]) / (p2[1] - p1[1])
            return (p1[0] + t * (p2[0] - p1[0]), y)

        poly = clip_edge(poly, lambda p: p[0] >= x_min, lambda s, e: inter_vert(s, e, x_min))
        poly = clip_edge(poly, lambda p: p[0] <= x_max, lambda s, e: inter_vert(s, e, x_max))
        poly = clip_edge(poly, lambda p: p[1] >= y_min, lambda s, e: inter_horiz(s, e, y_min))
        poly = clip_edge(poly, lambda p: p[1] <= y_max, lambda s, e: inter_horiz(s, e, y_max))
        return poly

    def polygon_area(poly) -> float:
        if len(poly) < 3:
            return 0.0
        area = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    arr = np.array(img)
    orig_w, orig_h = img.size
    # Match the resolution used in ``spot_has_car`` for consistency
    model_w, model_h = 1920, 1080
    scale_x = 1.0
    scale_y = 1.0

    input_arr = arr
    if orig_w > model_w or orig_h > model_h:
        scale_x = orig_w / model_w
        scale_y = orig_h / model_h
        resized = img.resize((model_w, model_h))
        input_arr = np.array(resized)

    results = model(input_arr, classes=VEHICLE_CLASSES)
    if not results or not results[0].boxes:
        return False

    hit_box: tuple[float, float, float, float] | None = None
    inter_box: tuple[float, float, float, float] | None = None
    for x1, y1, x2, y2 in results[0].boxes.xyxy.tolist():
        x1 *= scale_x
        y1 *= scale_y
        x2 *= scale_x
        y2 *= scale_y
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if not point_in_polygon(cx, cy, spot_poly):
            continue
        img2 = Image.open(io.BytesIO(image)).convert("RGB")
        frame = (
            cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
            if cv2 is not None
            else np.array(img2)
        )
        im_crop_path =crop_and_save_car(frame, int(x1),  int(y1),  int(x2),  int(y2))
        inter_poly = clip_polygon_rect(spot_poly, (x1, y1, x2, y2))
        inter_area = polygon_area(inter_poly)
        car_area = max((x2 - x1) * (y2 - y1), 1e-6)
        if inter_area / car_area >= 0.65:
            if same_entired_car(ticket_id,im_crop_path):
                hit_box = (x1, y1, x2, y2)
                if inter_poly:
                    ix = [p[0] for p in inter_poly]
                    iy = [p[1] for p in inter_poly]
                    inter_box = (min(ix), min(iy), max(ix), max(iy))
                break      
    if save_folder and cv2 is not None:
        try:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            points = np.array(spot_poly, np.int32)
            # Reshape to match cv2.polylines requirement
            points = points.reshape((-1, 1, 2))

            # Draw the polygon outline
            cv2.polylines(img_bgr, [points], isClosed=True, color=(255, 255, 0), thickness=2)
            # cv2.rectangle(
            #     img_bgr,
            #     (int(left), int(top)),
            #     (int(right), int(bottom)),
            #     (255, 255, 0),
            #     2,
            # )
            if results and results[0].boxes:
                for bx in results[0].boxes.xyxy.tolist():
                    bx = [bx[0] * scale_x, bx[1] * scale_y, bx[2] * scale_x, bx[3] * scale_y]
                    color = (0, 0, 255)
                    if hit_box and tuple(bx) == tuple(hit_box):
                        color = (0, 255, 0)
                    cv2.rectangle(
                        img_bgr,
                        (int(bx[0]), int(bx[1])),
                        (int(bx[2]), int(bx[3])),
                        color,
                        2,
                    )
            if inter_box:
                cv2.rectangle(
                    img_bgr,
                    (int(inter_box[0]), int(inter_box[1])),
                    (int(inter_box[2]), int(inter_box[3])),
                    (255, 0, 255),
                    2,
                )
            os.makedirs(save_folder, exist_ok=True)
            debug_path = os.path.join(
                save_folder,
                f"spot_check_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg",
            )
            cv2.imwrite(debug_path, img_bgr)
        except Exception:
            logger.error("Failed to save spot_has_car debug image", exc_info=True)

    return hit_box is not None

def process_plate_and_issue_ticket(
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
    *,
    request_timeout: float = 10.0,
    request_max_retries: int = 2,
):
    """
    1) Re-open saved snapshot, annotate & crop the parking region.
    2) Compare new main_crop vs. saved 'last' image for this spot; if same, skip.
    3) Otherwise, run YOLO→OCR, insert into plate_logs,
       and create Ticket (READ) or Ticket+ManualReview+clip thread (UNREAD),
       ensuring no duplicate open ticket per spot.
       Clip window: 8 seconds before to 8 seconds after trigger.
    """
    db_session = SessionLocal()
    try:
        # 1) Re-open snapshot and draw parking polygon
        snapshot_candidates = [
            os.path.join(park_folder, f)
            for f in os.listdir(park_folder)
            if f.startswith("snapshot") and f.lower().endswith(".jpg")
        ]

        if not snapshot_candidates:
            logger.error(
                "Snapshot missing: no 'snapshot*.jpg' found in %s", park_folder
            )
            return

        snapshot_path = snapshot_candidates[0]

        img = Image.open(snapshot_path)
        draw = ImageDraw.Draw(img)

        spot = (
            db_session.query(Spot)
            .filter_by(camera_id=camera_id, spot_number=spot_number)
            .first()
        )
        if spot is None:
            logger.error(
                "Spot %d on camera %d not found in DB", spot_number, camera_id
            )
            return

        spot.status = 1

        left, top, right, bottom = spot.bbox
        rand2=random.randint(111111, 999999999999)
        annotated_path = os.path.join(park_folder, f"annotated_{rand2}.jpg")
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        img.save(annotated_path)
        final_image_full = annotated_path
        main_crop = img.crop((left, top, right, bottom))
        rand3=random.randint(111111, 999999999999)
        main_crop_path = os.path.join(
            park_folder,
            f"main_crop_{payload['parking_area']}_{rand3}.jpg"
        )
        main_crop.save(main_crop_path)

        # Detect car within the spot and save a cropped version for comparison
        model = _get_plate_model()
        results = None
        if model is not None:
            arr = np.array(main_crop)
            results = model(arr, classes=VEHICLE_CLASSES)
        rand4=random.randint(111111, 999999999999)
        car_crop_path = os.path.join(park_folder, f"car_crop_{rand4}.jpg")
        car_crop = None
        if results and results[0].boxes:
            x1p, y1p, x2p, y2p = results[0].boxes.xyxy[0].tolist()
            x1i, y1i, x2i, y2i = map(int, (x1p, y1p, x2p, y2p))
            car_crop = main_crop.crop((x1i, y1i, x2i, y2i))
            car_crop.save(car_crop_path)

        # 2) Feature-match vs. last image for this spot
        spot_key = f"spot_{camera_id}_{spot_number}.jpg"
        last_image_path = os.path.join(SPOT_LAST_DIR, spot_key)

        # if car_crop and os.path.isfile(last_image_path):
        #     try:
        #         same = is_same_car(
        #             last_image_path,
        #             car_crop_path,
        #         )
        #         if same:
        #             logger.debug(
        #                 "Spot %d camera %d: same car detected → skip OCR/ticket",
        #                 spot_number, camera_id
        #             )
        #             return
        #     except Exception:
        #         logger.error("Error in feature-matching", exc_info=True)

        # Overwrite last-seen image with the cropped car
        try:
            if car_crop:
                shutil.copy(car_crop_path, last_image_path)
            else:
                shutil.copy(snapshot_path, last_image_path)
        except Exception:
            logger.error("Failed to update last-seen image", exc_info=True)

        # 3) Use detection results on main_crop to locate the license plate

        plate_status = "UNREAD"
        plate_number = None
        plate_code   = None
        plate_city   = None
        conf_val     = None

        if results and results[0].boxes:
            x1p, y1p, x2p, y2p = results[0].boxes.xyxy[0].tolist()
            x1i, y1i, x2i, y2i = map(int, (x1p, y1p, x2p, y2p))
            plate_crop = main_crop.crop((x1i, y1i, x2i, y2i))

            # Enhance crop before sending to OCR
            try:
                arr_bgr = cv2.cvtColor(np.array(plate_crop), cv2.COLOR_RGB2BGR)
                # arr_bgr = enhance_image_array(arr_bgr)
                plate_crop = Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))
            except Exception:
                logger.error("Plate enhancement failed", exc_info=True)
            rand5=random.randint(111111, 999999999999)
            tmp_candidate_path = os.path.join(park_folder, f"plate_candidate_{rand5}.jpg")
            plate_crop.save(tmp_candidate_path)

            # 4) Base64-encode plate crop and send to OCR
            with open(tmp_candidate_path, "rb") as f:
                plate_bytes = f.read()
            plate_b64 = base64.b64encode(plate_bytes).decode("utf-8")

            ocr_payload  = {
                "token":  OCR_TOKEN,
                "base64": plate_b64,
                "pole_id": pole_id
            }
            ocr_response = send_request_with_retry(
                "https://parkonic.cloud/ParkonicJLT/anpr/engine/process",
                ocr_payload,
                timeout=request_timeout,
                max_retries=request_max_retries,
            )

            logger.debug(f"Raw OCR response: {ocr_response!r}")

            # double json.loads
            intermediate = None
            if isinstance(ocr_response, str):
                try:
                    intermediate = json.loads(ocr_response)
                    logger.debug("After first json.loads: %s", type(intermediate).__name__)
                except Exception:
                    logger.error("First json.loads failed", exc_info=True)
                    plate_status = "UNREAD"
            else:
                logger.debug("OCR response not str → UNREAD")
                plate_status = "UNREAD"

            ocr_json = None
            if isinstance(intermediate, str):
                try:
                    ocr_json = json.loads(intermediate)
                    logger.debug("After second json.loads → dict")
                except Exception:
                    logger.error("Second json.loads failed", exc_info=True)
                    plate_status = "UNREAD"
            elif isinstance(intermediate, dict):
                ocr_json = intermediate
                logger.debug("OCR JSON is dict")
            else:
                logger.error("Unexpected OCR intermediate type: %s", type(intermediate).__name__)
                plate_status = "UNREAD"

            if isinstance(ocr_json, dict):
                try:
                    confidance_value = int(ocr_json.get("confidance", 0))
                    logger.debug("OCR confidance: %d", confidance_value)
                    if confidance_value >= 50:
                        plate_status = "READ"
                        plate_number = ocr_json.get("text", "")
                        plate_code   = ocr_json.get("category", "")
                        city_code    = ocr_json.get("cityName", "")
                        conf_val     = confidance_value

                        plate_city = city_code
                    else:
                        logger.debug("Confidence %d < 5 → UNREAD", confidance_value)
                        plate_status = "UNREAD"

                except Exception:
                    logger.error("Failed to extract from ocr_json", exc_info=True)
                    plate_status = "UNREAD"

        # Fallback: reuse request-provided snapshot data rather than fetching
        # a new frame from the camera. Some deployments cannot access the
        # camera directly, so we rely solely on the base64 snapshot that
        # accompanied the entry trigger.
        if plate_status == "UNREAD":
            try:
                img = None
                retry_snapshot = snapshot_path
                snapshot_bytes = None
                snapshot_b64 = payload.get("snapshot")
                if isinstance(snapshot_b64, str) and snapshot_b64:
                    try:
                        snapshot_bytes = base64.b64decode(snapshot_b64)
                    except Exception:
                        logger.error("Retry snapshot decode failed", exc_info=True)

                if snapshot_bytes is not None:
                    rand6 = random.randint(111111, 999999999999)
                    retry_snapshot = os.path.join(
                        park_folder, f"retry_snapshot_{rand6}.jpg"
                    )
                    with open(retry_snapshot, "wb") as f:
                        f.write(snapshot_bytes)
                    img = Image.open(io.BytesIO(snapshot_bytes)).convert("RGB")
                else:
                    rand6 = random.randint(111111, 999999999999)
                    candidate = os.path.join(
                        park_folder, f"retry_snapshot_{rand6}.jpg"
                    )
                    try:
                        shutil.copy(snapshot_path, candidate)
                        retry_snapshot = candidate
                    except Exception:
                        retry_snapshot = snapshot_path
                    img = Image.open(retry_snapshot).convert("RGB")

                draw = ImageDraw.Draw(img)
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                rand7 = random.randint(111111, 999999999999)
                annotated_path = os.path.join(
                    park_folder, f"annotated_retry_{rand7}.jpg"
                )
                img.save(annotated_path)
                final_image_full = annotated_path

                vehicle_box = (
                    int(payload.get("vehicle_frame_x1", left)),
                    int(payload.get("vehicle_frame_y1", top)),
                    int(payload.get("vehicle_frame_x2", right)),
                    int(payload.get("vehicle_frame_y2", bottom)),
                )
                width, height = img.size
                vx1 = max(0, min(width, vehicle_box[0]))
                vy1 = max(0, min(height, vehicle_box[1]))
                vx2 = max(0, min(width, vehicle_box[2]))
                vy2 = max(0, min(height, vehicle_box[3]))
                if vx1 >= vx2 or vy1 >= vy2:
                    vx1, vy1, vx2, vy2 = left, top, right, bottom

                main_crop = img.crop((vx1, vy1, vx2, vy2))
                rand8 = random.randint(111111, 999999999999)
                main_crop_path = os.path.join(
                    park_folder, f"main_crop_retry_{rand8}.jpg"
                )
                main_crop.save(main_crop_path)

                model = _get_plate_model()
                results = None
                if model is not None:
                    arr = np.array(main_crop)
                    results = model(arr, classes=VEHICLE_CLASSES)
                if results and results[0].boxes:
                    x1p, y1p, x2p, y2p = results[0].boxes.xyxy[0].tolist()
                    x1i, y1i, x2i, y2i = map(int, (x1p, y1p, x2p, y2p))
                    plate_crop = main_crop.crop((x1i, y1i, x2i, y2i))

                    if cv2 is not None:
                        try:
                            arr_bgr = cv2.cvtColor(
                                np.array(plate_crop), cv2.COLOR_RGB2BGR
                            )
                            arr_bgr = enhance_image_array(arr_bgr)
                            plate_crop = Image.fromarray(
                                cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
                            )
                        except Exception:
                            logger.error("Plate enhancement failed", exc_info=True)
                    rand9=random.randint(111111, 999999999999)
                    tmp_candidate_path = os.path.join(park_folder, f"plate_candidate_retry_{rand9}.jpg")
                    plate_crop.save(tmp_candidate_path)

                    with open(tmp_candidate_path, "rb") as f:
                        plate_bytes = f.read()
                    plate_b64 = base64.b64encode(plate_bytes).decode("utf-8")
                    ocr_payload = {
                        "token": OCR_TOKEN,
                        "base64": plate_b64,
                        "pole_id": pole_id,
                    }
                    ocr_response = send_request_with_retry(
                        "https://parkonic.cloud/ParkonicJLT/anpr/engine/process",
                        ocr_payload,
                        timeout=request_timeout,
                        max_retries=request_max_retries,
                    )

                    logger.debug(f"Retry OCR response: {ocr_response!r}")

                    intermediate = None
                    if isinstance(ocr_response, str):
                        try:
                            intermediate = json.loads(ocr_response)
                        except Exception:
                            logger.error("First json.loads failed on retry", exc_info=True)
                            intermediate = None
                    if isinstance(intermediate, str):
                        try:
                            ocr_json = json.loads(intermediate)
                        except Exception:
                            logger.error("Second json.loads failed on retry", exc_info=True)
                            ocr_json = None
                    elif isinstance(intermediate, dict):
                        ocr_json = intermediate
                    else:
                        ocr_json = None

                    if isinstance(ocr_json, dict):
                        try:
                            confidance_value = int(ocr_json.get("confidance", 0))
                            if confidance_value >= 50:
                                plate_status = "READ"
                                plate_number = ocr_json.get("text", "")
                                plate_code = ocr_json.get("category", "")
                                plate_city = ocr_json.get("cityName", "")
                                conf_val = confidance_value
                                # city_map = {
                                #     "AE-AZ": "Abu Dhabi",
                                #     "AE-DU": "Dubai",
                                #     "AE-SH": "Sharjah",
                                #     "AE-AJ": "Ajman",
                                #     "AE-RK": "RAK",
                                #     "AE-FU": "Fujairah",
                                #     "AE-UQ": "UAQ",
                                # }
                                # plate_city = city_map.get(city_code, "Unknown")
                                
                        except Exception:
                            logger.error("Failed to extract from retry ocr_json", exc_info=True)
            except Exception:
                logger.error("Retry capture or OCR failed", exc_info=True)

        # 5) Save final plate image
        os.makedirs(PLATES_READ_DIR,   exist_ok=True)
        os.makedirs(PLATES_UNREAD_DIR, exist_ok=True)

        micro = datetime.utcnow().strftime('%f')
        rand10=random.randint(111111, 999999999999)
        final_plate_filename = f"{camera_id}_{rand10}_{micro}.jpg"
        dest_dir = PLATES_READ_DIR if plate_status == "READ" else PLATES_UNREAD_DIR
        final_plate_path = os.path.join(dest_dir, final_plate_filename)

        if "tmp_candidate_path" in locals() and os.path.exists(tmp_candidate_path):
            shutil.copy(tmp_candidate_path, final_plate_path)
        else:
            shutil.copy(main_crop_path, final_plate_path)

        ticket_image_b64 = None
        final_image_full_bas64 = None
        
        try:
            with open(final_plate_path, "rb") as f:
                ticket_image_b64 = base64.b64encode(f.read()).decode("utf-8")
            with open(final_image_full, "rb") as s:
                final_image_full_bas64 = base64.b64encode(s.read()).decode("utf-8")
        except Exception:
            logger.error("Failed to read final plate image for ticket", exc_info=True)

        # 6) Insert into plate_logs
        new_plate_log = PlateLog(
            camera_id    = camera_id,
            car_id       = payload.get("car_id"),
            plate_number = plate_number,
            plate_code   = plate_code,
            plate_city   = plate_city,
            confidence   = conf_val,
            image_path   = final_plate_path,
            status       = plate_status,
            attempt_ts   = datetime.utcnow()
        )
        db_session.add(new_plate_log)
        db_session.commit()
        logger.debug("Inserted into plate_logs: camera_id=%d, status=%s", camera_id, plate_status)

        # 6b) Insert an entry in manual_reviews to keep track of the processed
        # plate image and snapshot directory for debugging.
        new_review_tx = ManualReview(
            camera_id       = camera_id,
            spot_number     = spot_number,
            event_time      = datetime.fromisoformat(payload["time"]),
            image_path      = final_plate_path,
            plate_status    = plate_status,
            plate_image     = final_plate_filename,
            snapshot_folder = os.path.basename(park_folder),
            review_status   = "RESOLVED" if plate_status == "READ" else "PENDING"
        )
        db_session.add(new_review_tx)
        db_session.flush()
        review_id = new_review_tx.id
        db_session.commit()

        # 7) If READ → create Ticket or reopen existing one
        if plate_status == "READ":
            try:
                existing_closed = (
                    db_session.query(Ticket)
                    .filter(
                        Ticket.camera_id == camera_id,
                        Ticket.spot_number == spot_number,
                    )
                    .order_by(Ticket.entry_time.desc())
                    .first()
                )
                if (
                    existing_closed
                    and existing_closed.exit_time is not None
                    and (existing_closed.plate_number or "") == (plate_number or "")
                    and (existing_closed.plate_code or "") == (plate_code or "")
                    and (existing_closed.plate_city or "") == (plate_city or "")
                ):
                    existing_closed.exit_time = None
                    existing_closed.exit_clip_path = None
                    db_session.commit()
                    logger.debug(
                        "Reopened ticket id=%d for same car", existing_closed.id
                    )
                    return
            except Exception:
                logger.error("Reopen ticket check failed", exc_info=True)
            if ticket_image_b64:
                img_list = [ticket_image_b64,ticket_image_b64]
            else:
                img_list = []
                try:
                    with open(annotated_path, "rb") as f:
                        img_list.append(base64.b64encode(f.read()).decode("utf-8"))
                except Exception:
                    logger.error("Failed to read annotated image for API", exc_info=True)
                try:
                    with open(main_crop_path, "rb") as f:
                        img_list.append(base64.b64encode(f.read()).decode("utf-8"))
                except Exception:
                    logger.error("Failed to read cropped image for API", exc_info=True)
                if not img_list:
                    with open(final_plate_path, "rb") as f:
                        img_list = [base64.b64encode(f.read()).decode("utf-8")]

            try:
                new_ticket = Ticket(
                    camera_id        = camera_id,
                    spot_number      = spot_number,
                    plate_number     = plate_number,
                    plate_code       = plate_code,
                    plate_city       = plate_city,
                    confidence       = conf_val,
                    entry_time       = datetime.fromisoformat(payload["time"]),
                    parkonic_trip_id = None,
                    image_base64     = ticket_image_b64,
                    entry_image_path = final_image_full,
                )
                db_session.add(new_ticket)
                db_session.commit()
                new_review_tx.ticket_id = new_ticket.id
                db_session.commit()
                logger.debug("Inserted ticket id=%d", new_ticket.id)
            except Exception:
                logger.error("Ticket INSERT failed", exc_info=True)

        # 8) If UNREAD → create Ticket + ManualReview + spawn clip thread,
        #    but only if no existing open ticket for this spot
        elif plate_status == "UNREAD":
            try:
                # a) Check if an open ticket exists (exit_time is NULL)
                existing_ticket = db_session.query(Ticket).filter_by(
                    camera_id   = camera_id,
                    spot_number = spot_number,
                    exit_time   = None
                ).first()

                if existing_ticket:
                    logger.debug(
                        "Spot %d on camera %d already has open ticket (id=%d) → skip new ticket/manual review",
                        spot_number, camera_id, existing_ticket.id
                    )
                    return

                # The manual review entry was already created above as
                # ``new_review_tx`` with review_status=PENDING. Reuse it here.
                logger.debug("Using existing manual_review id=%d for UNREAD plate", review_id)

                # c) INSERT a Ticket for UNREAD plate
                new_ticket = Ticket(
                    camera_id        = camera_id,
                    spot_number      = spot_number,
                    plate_number     = plate_number or "UNKNOWN",
                    plate_code       = plate_code or "",
                    plate_city       = plate_city or "",
                    confidence       = conf_val or 0,
                    entry_time       = datetime.fromisoformat(payload["time"]),
                    parkonic_trip_id = None,
                    image_base64     = ticket_image_b64,
                    entry_image_path = final_image_full
                )
                db_session.add(new_ticket)
                db_session.commit()
                logger.debug("Inserted UNREAD ticket into tickets: id=%d", new_ticket.id)

                # link manual review to the new ticket
                new_review_tx.ticket_id = new_ticket.id
                db_session.commit()

                # d) Fetch camera clip for manual review in the current background task
                def fetch_and_update_clip(rid: int, cam_ip: str, user: str, pwd: str, ev_time: datetime):
                    start_dt = ev_time - timedelta(seconds=15)
                    end_dt   = ev_time + timedelta(seconds=5)
                    clip_path = request_camera_clip(
                        camera_ip    = cam_ip,
                        username     = user,
                        password     = pwd,
                        start_dt     = start_dt,
                        end_dt       = end_dt,
                        segment_name = ev_time.strftime("%Y%m%d%H%M%S"),
                        unique_tag   = str(rid),
                    )
                    session_t = SessionLocal()
                    try:
                        if clip_path:
                            review_obj = session_t.query(ManualReview).get(rid)
                            if review_obj:
                                review_obj.clip_path = clip_path
                                session_t.commit()
                                logger.debug(
                                    "Updated manual_reviews.clip_path=%s for id=%d", clip_path, rid
                                )

                                # Gather additional info for API
                                crop_objs = session_t.query(CropZone).filter(CropZone.camera_id == review_obj.camera_id).all()
                                crop_list = [c.points for c in crop_objs]
                                spot_obj = (
                                    session_t.query(Spot)
                                    .filter_by(camera_id=review_obj.camera_id, spot_number=review_obj.spot_number)
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
                                        review_id=review_obj.id,
                                        camera_id=review_obj.camera_id,
                                        spot_number=review_obj.spot_number,
                                        crop_zones=crop_list,
                                        spot_details=spot_details,
                                        video_path=clip_path,
                                        server_base="http://10.11.5.100:18007",
                                    )
                                except Exception:
                                    logger.error("send_review_video failed", exc_info=True)
                        else:
                            logger.error("Could not obtain clip for manual review id=%d", rid)
                    except Exception:
                        logger.error("Exception in fetch_and_update_clip", exc_info=True)
                        session_t.rollback()
                    finally:
                        session_t.close()

                fetch_and_update_clip(
                    review_id,
                    camera_ip,
                    camera_user,
                    camera_pass,
                    datetime.fromisoformat(payload["time"]),
                )

            except Exception:
                logger.error("manual_reviews INSERT failed", exc_info=True)
                db_session.rollback()

    except Exception:
        logger.error("process_plate_and_issue_ticket exception", exc_info=True)
        db_session.rollback()
    finally:
        db_session.close()
