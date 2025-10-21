"""Core parking logic utilities with optional CV/ML dependencies.

The production code relies on Ultralytics YOLO and OpenCV, both of which pull in
native libraries that are unavailable in the execution environment used for the
kata.  Importing this module previously failed immediately because ``cv2`` could
not be initialised which prevented the FastAPI application – and therefore the
tests – from starting.  The heavy dependencies are now loaded lazily and all
operations guard against them being absent so the rest of the application can be
exercised without GPU support.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from typing import Optional

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

from config import OCR_TOKEN, VEHICLE_CLASSES, YOLO_MODEL_PATH
from db import SessionLocal
from logger import logger
from models import Spot, Ticket
from network import send_request_with_retry
from utils import is_same_car

AN_VIDEO_OUTPUT_DIR = "AN_VIDEO_OUTPUT_DIR"
CAR_CROPS = "car_crops"

_yolo_model: Optional[object] = None
_yolo_available: Optional[bool] = None


def _ensure_yolo_model() -> bool:
    """Load the Ultralytics YOLO model when the dependencies are present."""

    global _yolo_model, _yolo_available

    if _yolo_available is not None:
        return _yolo_available

    if YOLO is None:  # pragma: no cover - optional dependency
        logger.warning("Ultralytics YOLO not available; skipping video analysis")
        _yolo_available = False
        return False

    try:  # pragma: no cover - heavy optional dependency
        model = YOLO(YOLO_MODEL_PATH)
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        model.to(device)
    except Exception as exc:
        logger.warning("Failed to initialise YOLO model: %s", exc)
        _yolo_available = False
        return False

    _yolo_model = model
    _yolo_available = True
    return True


def crop_and_save_car(frame, x1, y1, x2, y2, prefix="car") -> str:
    """Crop car from frame and save it. Returns saved file path."""
    crop = frame[y1:y2, x1:x2]
    filename = f"{prefix}_{uuid.uuid4()}.jpg"
    path = os.path.join(CAR_CROPS, filename)
    os.makedirs("car_crops", exist_ok=True)
    if cv2 is not None:  # pragma: no branch - runtime guard
        cv2.imwrite(path, crop)
    else:
        Image.fromarray(crop).save(path)
    return path


def same_entired_car(
    ticket_id: int = 1,
    exit_crop_path: str = "",
):
    db = SessionLocal()
    try:
        ticket = db.query(Ticket).filter_by(id=ticket_id).first()
    finally:
        db.close()
    
    
    filename = f"car_pic_{uuid.uuid4()}.jpg"
    entry_crop_path = os.path.join(CAR_CROPS, filename)
    logger.info("entered image path: %s", entry_crop_path)
    img_data = base64.b64decode(ticket.image_base64)
    
    with open(entry_crop_path, "wb") as f:
        f.write(img_data)
    if ticket is None or not os.path.isfile(entry_crop_path):
        logger.error("ticket or entry image ", exc_info=True)
        return False
    # Attempt OCR comparison first using the exit image
    ocr_res = ocr_plate_from_image(exit_crop_path) if exit_crop_path else None
    if ocr_res:
        try:
            plate_match = (
                ocr_res.get("number", "").upper()
                == (ticket.plate_number or "").upper()
                and ocr_res.get("code", "").upper()
                == (ticket.plate_code or "").upper()
                and ocr_res.get("city", "").upper()
                == (ticket.plate_city or "").upper()
            )
            if plate_match:
                logger.debug("✅ OCR plate matches ticket details.")
                return True
        except Exception:
            logger.error("Error comparing OCR result", exc_info=True)
    logger.debug("🚗 Found car inside the polygon in entry frame.")
    match = is_same_car(entry_crop_path, exit_crop_path)
    logger.debug(
        f"🔍 Compare entry to exit car: {'✅ SAME' if match else '❌ DIFFERENT'}"
    )
    return match


def ocr_plate_from_image(
    image_path: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 2,
):
    """Return OCR result dict for the given image or ``None`` on failure."""
    try:
        with open(image_path, "rb") as f:
            plate_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {"token": OCR_TOKEN, "base64": plate_b64}
        resp = send_request_with_retry(
            "https://parkonic.cloud/ParkonicJLT/anpr/engine/process",
            payload,
            timeout=timeout,
            max_retries=max_retries,
        )

        intermediate = resp
        if isinstance(intermediate, str):
            try:
                intermediate = json.loads(intermediate)
            except Exception:
                pass
        if isinstance(intermediate, str):
            try:
                data = json.loads(intermediate)
            except Exception:
                data = None
        elif isinstance(intermediate, dict):
            data = intermediate
        else:
            data = None

        if isinstance(data, dict):
            try:
                conf = int(data.get("confidance", 0))
                if conf >= 50:
                    return {
                        "number": data.get("text", ""),
                        "code": data.get("category", ""),
                        "city": data.get("cityName", ""),
                        "confidence": conf,
                    }
            except Exception:
                logger.error("Failed to parse OCR response", exc_info=True)
    except Exception:
        logger.error("OCR request failed", exc_info=True)
    return None


def exit_video_analyses(
    video_path: str,
    camera_id: int,
    spot_number: int,
    ticket_id: int = 1,
    model_input_size: tuple[int, int] = (640, 384),
):

    if cv2 is None:
        logger.error("OpenCV is required for exit video analysis")
        return False

    if not _ensure_yolo_model():
        return False

    model = _yolo_model
    if model is None:
        return False

    if not os.path.isfile(video_path):
        logger.error("Exit video is unavailable", exc_info=True)
        return False
    db = SessionLocal()
    try:
        spot = (
            db.query(Spot)
            .filter_by(camera_id=camera_id, spot_number=spot_number)
            .first()
        )
        ticket = db.query(Ticket).filter_by(id=ticket_id).first()
    finally:
        db.close()

    if spot is None or ticket is None:
        return False

    spot_poly = [
        (float(spot.p1_x), float(spot.p1_y)),
        (float(spot.p2_x), float(spot.p2_y)),
        (float(spot.p3_x), float(spot.p3_y)),
        (float(spot.p4_x), float(spot.p4_y)),
    ]

    # Create polygon
    polygon = Polygon(spot_poly)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Exit video is unavailable", exc_info=True)
        return False
    unique_filename = f"{ticket_id}_{spot_number}_{camera_id}_{uuid.uuid4()}.mp4"
    output_path = os.path.join(AN_VIDEO_OUTPUT_DIR, unique_filename)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    scale_x = orig_w / model_input_size[0]
    scale_y = orig_h / model_input_size[1]

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h)
    )

    previous_status = "empty"
    reference_image = None
    mismatch_detected = False
    car_has_exited = False  # New flag: track if car exited
    replacement_matched = False  # New flag: track if same car re-entered
    mismatch_entry_detected = False
    multiple_cars_detected = False  # New flag: track if another car entered
    ocr_plate_matched = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, model_input_size)
        results = model.track(
            source=resized,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
             verbose=False
        )[0]
        frame_copy = frame
        # Draw polygon
        cv2.polylines(frame, [np.array(spot_poly, np.int32)], True, (255, 255, 0), 2)

        car_found = False
        current_car_bbox = None
        cars_inside = []

        for box in results.boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            orig_cx = int(cx * scale_x)
            orig_cy = int(cy * scale_y)
            orig_x1 = int(x1 * scale_x)
            orig_y1 = int(y1 * scale_y)
            orig_x2 = int(x2 * scale_x)
            orig_y2 = int(y2 * scale_y)

            point = Point(orig_cx, orig_cy)
            is_inside = polygon.contains(point)
            current_status = "IN" if is_inside else "OUT"

            # 🔷 Draw analysis
            color = (0, 255, 0) if is_inside else (0, 0, 255)
            label = f"ID {track_id}"
            cv2.rectangle(frame, (orig_x1, orig_y1), (orig_x2, orig_y2), color, 2)
            cv2.putText(
                frame,
                label,
                (orig_x1, orig_y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            cv2.circle(frame, (orig_cx, orig_cy), 4, color, -1)

            if is_inside:
                cars_inside.append((orig_x1, orig_y1, orig_x2, orig_y2))

        if len(cars_inside) > 0:
            car_found = True
            current_car_bbox = cars_inside[0]
            if len(cars_inside) > 1:
                multiple_cars_detected = True

        current_status = "occupied" if car_found else "empty"

        # Detect empty → occupied
        if previous_status == "empty" and current_status == "occupied":
            logger.debug("🟢 Spot became occupied.")

            if current_car_bbox:
                x1, y1, x2, y2 = current_car_bbox
                new_image = crop_and_save_car(frame_copy, x1, y1, x2, y2, prefix="car")
                plate_res = ocr_plate_from_image(new_image)
                plate_match = False
                if plate_res:
                    try:
                        plate_match = (
                            plate_res.get("number", "").upper() == (ticket.plate_number or "").upper()
                            and plate_res.get("code", "").upper() == (ticket.plate_code or "").upper()
                            and plate_res.get("city", "").upper() == (ticket.plate_city or "").upper()
                        )
                    except Exception:
                        plate_match = False

                if plate_match:
                    logger.debug("✅ OCR plate matches ticket details.")
                    ocr_plate_matched = True
                    if reference_image is None:
                        reference_image = new_image
                        logger.debug("📸 First car saved as reference.")
                else:
                    if not same_entired_car(ticket_id, new_image):
                        mismatch_entry_detected = True
                    if reference_image is None:
                        reference_image = new_image
                        logger.debug("📸 First car saved as reference.")
                    else:
                        is_same = is_same_car(reference_image, new_image)
                        logger.debug(
                            f"🔍 Compare to reference: {'✅ SAME' if is_same else '❌ DIFFERENT'}"
                        )
                        if not is_same:
                            mismatch_detected = True
                        else:
                            replacement_matched = True  # same car re-entered

        # Detect occupied → empty
        if previous_status == "occupied" and current_status == "empty":
            logger.debug("🔴 Car exited the parking.")
            car_has_exited = True

        previous_status = current_status
        out.write(frame)

    cap.release()
    out.release()

    final_status = previous_status
    spot_empty_after_exit = final_status == "empty"

    if multiple_cars_detected:
        logger.debug(
            "❌ Another car entered while one was already parked.ticket id =%d",
            ticket_id,
        )
        return 6

    if ocr_plate_matched:
        if car_has_exited and spot_empty_after_exit:
            logger.debug(
                "✅ OCR matched, car exited and spot empty.ticket id =%d",
                ticket_id,
            )
            return 2
        logger.debug(
            "⚠️ OCR matched but car did not exit cleanly.ticket id =%d",
            ticket_id,
        )
        return 3

    if mismatch_entry_detected:
        if spot_empty_after_exit:
            logger.debug(
                "❌ diffrent car exited after the car exit, spot empty.ticket id =%d",
                ticket_id,
            )
            return 5
        logger.debug("❌ diffrent car is exiting.ticket id =%d", ticket_id)
        return 0

    # Final evaluation
    if mismatch_detected:
        logger.debug("❌ Car was replaced by a different one. ticket id =%d", ticket_id)
        return 1
    if car_has_exited and not replacement_matched:
        logger.debug("✅ Car exited and was not replaced.ticket id =%d", ticket_id)
        return 2
    if reference_image:
        logger.debug(
            "✅ The same car is still occupying the spot or re-entered.ticket id =%d",
            ticket_id,
        )
        return 3
    logger.debug("⚠️ Spot was never occupied.ticket id =%d", ticket_id)
    return 4


