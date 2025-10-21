"""Utility helpers used throughout the StreetServer2 service.

The original implementation of :func:`is_same_car` relied on the ``torchreid``
project which pulls in OpenCV with GUI backends. The execution environment used
in the kata does not provide those native dependencies which caused the module
import to fail before the FastAPI app – and the tests – could even start.  This
module now loads the heavy dependencies lazily and provides a lightweight numpy
fallback so that unit tests can run without GPU, OpenCV or torchreid being
installed.

An ``is_same_image`` helper has also been introduced.  It compares two images by
focusing solely on the parking spot polygon stored in the database, allowing the
tests to verify that unrelated parts of the frame (for example logos or
timestamp overlays) do not influence the similarity check.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    F = None  # type: ignore

from config import OS_NET_MODEL
from db import SessionLocal
from models import Spot

logger = logging.getLogger(__name__)


# === SIZE CONSTANTS ===
DISPLAY_IMG_HEIGHT = 300    # px
PADDING = 20                # px
TEXT_AREA_HEIGHT = 50       # px
FONT_SIZE = 40              # scalable TrueType font size

# === OUTPUT FOLDER ===
OUTPUT_DIR = "check_similarity"
trained_weights = OS_NET_MODEL
height, width = 256, 128   # model input size

_model = None
test_transform = None
_reid_available: bool | None = None


def _ensure_output_dir() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def _ensure_reid_model() -> bool:
    """Load the torchreid model on first use.

    Returns ``True`` when the model is ready.  When the optional dependencies
    are missing we simply fall back to a numpy based comparison instead of
    raising during import time which keeps the API usable in minimal
    environments.
    """

    global _model, test_transform, _reid_available

    if _reid_available is not None:
        return _reid_available

    if torch is None:  # pragma: no cover - exercised only when torch missing
        logger.warning("torch not available – falling back to numpy comparison")
        _reid_available = False
        return False

    try:  # pragma: no cover - heavy optional dependency
        from torchreid.reid.models import build_model
        from torchreid.reid.data.transforms import build_transforms
    except Exception as exc:  # pragma: no cover - torchreid optional
        logger.warning("torchreid not available: %s", exc)
        _reid_available = False
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        name="osnet_x1_0", num_classes=1, loss="softmax", pretrained=False
    )
    try:
        sd = torch.load(trained_weights, map_location=device)
    except Exception as exc:  # pragma: no cover - weights optional in tests
        logger.warning("Failed to load OSNet weights: %s", exc)
        _reid_available = False
        return False

    sd.pop("classifier.weight", None)
    sd.pop("classifier.bias", None)
    model.load_state_dict(sd, strict=False)
    _model = model.to(device).eval()
    _, test_transform = build_transforms(height=height, width=width, is_train=False)
    _reid_available = True
    return True

# === FEATURE EXTRACTION ===
def _extract_feat(img_path: str):
    """Return a torch feature vector for the given image path."""

    if _model is None or test_transform is None or torch is None or F is None:
        raise RuntimeError("ReID model not initialised")

    img = Image.open(img_path).convert("RGB")
    device = next(_model.parameters()).device
    img_t = test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():  # pragma: no cover - requires torch
        feat = _model(img_t)
        feat = F.normalize(feat, p=2, dim=1)
    return feat.cpu()


def _numpy_cosine_similarity(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Compute cosine similarity for flattened numpy arrays."""

    v1 = arr1.astype(np.float32).ravel()
    v2 = arr2.astype(np.float32).ravel()
    if v1.size == 0 or v2.size == 0:
        return 0.0
    v1 -= v1.mean()
    v2 -= v2.mean()
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 1.0
    return float(np.dot(v1, v2) / denom)

# === IMAGE SAVING FUNCTION ===
def save_comparison_image(img1_path: str,
                          img2_path: str,
                          match: bool,
                          cos_sim: float) -> str:
    # load & resize inputs
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    w1 = int(img1.width * DISPLAY_IMG_HEIGHT / img1.height)
    w2 = int(img2.width * DISPLAY_IMG_HEIGHT / img2.height)
    img1 = img1.resize((w1, DISPLAY_IMG_HEIGHT), resample=Image.LANCZOS)
    img2 = img2.resize((w2, DISPLAY_IMG_HEIGHT), resample=Image.LANCZOS)

    # prepare canvas
    canvas_w = w1 + w2 + PADDING * 3
    canvas_h = DISPLAY_IMG_HEIGHT + TEXT_AREA_HEIGHT + PADDING * 2
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    canvas.paste(img1, (PADDING, PADDING))
    canvas.paste(img2, (PADDING + w1 + PADDING, PADDING))

    # draw text
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    verdict = "MATCH" if match else "NO MATCH"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    txt = f"{timestamp}   {verdict}   ({cos_sim:.4f})"
    l, t, r, b = draw.textbbox((0,0), txt, font=font)
    text_w, text_h = r - l, b - t
    x = (canvas_w - text_w) // 2
    y = DISPLAY_IMG_HEIGHT + PADDING + (TEXT_AREA_HEIGHT - text_h) // 2
    draw.text((x, y), txt, fill='black', font=font)

    # save
    fname_ts = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    fname = f"{fname_ts} {'true' if match else 'false'}.jpg"
    _ensure_output_dir()
    out_path = os.path.join(OUTPUT_DIR, fname)
    canvas.save(out_path, quality=90)
    return out_path

# === COMPARISON FUNCTION ===
def is_same_car(img1_path: str, img2_path: str, threshold: float = 0.8):
    cos_sim: float | None = None
    match: bool

    if _ensure_reid_model():  # pragma: no branch - runtime decision
        try:
            feat1 = _extract_feat(img1_path)
            feat2 = _extract_feat(img2_path)
            if F is not None:
                cos_sim = F.cosine_similarity(feat1, feat2).item()
        except Exception:  # pragma: no cover - requires optional deps
            logger.exception("Falling back to numpy comparison for car match")

    if cos_sim is None:
        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")
        cos_sim = _numpy_cosine_similarity(np.array(img1), np.array(img2))

    match = cos_sim > threshold
    try:
        save_comparison_image(img1_path, img2_path, match, cos_sim)
    except Exception:  # pragma: no cover - best effort only
        logger.debug("Unable to save comparison image", exc_info=True)

    return match


def _polygon_from_spot(spot: Spot) -> list[tuple[float, float]]:
    return [
        (float(spot.p1_x), float(spot.p1_y)),
        (float(spot.p2_x), float(spot.p2_y)),
        (float(spot.p3_x), float(spot.p3_y)),
        (float(spot.p4_x), float(spot.p4_y)),
    ]


def _build_mask(size: Sequence[int], polygon: Sequence[Sequence[float]]) -> np.ndarray:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(list(map(tuple, polygon)), outline=255, fill=255)
    arr = np.array(mask, dtype=np.float32) / 255.0
    return arr


def is_same_image(
    img1_path: str,
    img2_path: str,
    *,
    camera_id: int,
    spot_number: int,
    threshold: float = 0.95,
) -> bool:
    """Compare two images focusing on the configured parking spot polygon."""

    session = SessionLocal()
    try:
        spot = (
            session.query(Spot)
            .filter_by(camera_id=camera_id, spot_number=spot_number)
            .first()
        )
    finally:
        session.close()

    if spot is None:
        logger.debug(
            "Spot not found for camera %s spot %s", camera_id, spot_number
        )
        return False

    polygon = _polygon_from_spot(spot)

    img1 = Image.open(img1_path).convert("L")
    img2 = Image.open(img2_path).convert("L")
    if img2.size != img1.size:
        img2 = img2.resize(img1.size, Image.BILINEAR)

    mask = _build_mask(img1.size, polygon)
    mask_pixels = mask > 0.5
    if not np.any(mask_pixels):
        logger.debug("Spot polygon for camera %s has zero area", camera_id)
        return False

    arr1 = np.array(img1, dtype=np.float32)[mask_pixels]
    arr2 = np.array(img2, dtype=np.float32)[mask_pixels]

    if arr1.size == 0 or arr2.size == 0:
        return False

    similarity = _numpy_cosine_similarity(arr1, arr2)
    return similarity >= threshold

# # === EXAMPLE USAGE ===
# if __name__ == "__main__":
#     match = is_same_car("12.jpg", "22.jpg", threshold=0.6)
    
