# camera_clip.py

import time
import uuid
import requests
from requests.auth import HTTPDigestAuth
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
from datetime import datetime, timedelta
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from logger import logger
import os
# OpenCV's FFMPEG option expects a timeout in milliseconds. 240000 ms equals
# four minutes which gives the camera ample time to start streaming.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;400000"

def is_valid_mp4(path: str) -> bool:
    """Return True if the file at ``path`` can be opened and read as MP4."""
    if cv2 is None:
        return False
    if not os.path.isfile(path):
        return False
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return False
        ret, _ = cap.read()
        cap.release()
        if not ret:
            os.remove(path)
            return False
        return True
    except Exception:
        logger.error("Failed validating MP4 %s", path, exc_info=True)
        try:
            os.remove(path)
        except Exception:
            pass
        return False

VIDEO_CLIPS_DIR = "video_clips"
os.makedirs(VIDEO_CLIPS_DIR, exist_ok=True)

# Dedicated executor for camera network operations
CAMERA_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.environ.get("CAMERA_WORKERS", "4"))
)


def request_camera_clip(
    camera_ip: str,
    username: str,
    password: str,
    start_dt: datetime,
    end_dt: datetime,
    segment_name: str,
    unique_tag: str | None = None,
) -> Optional[str]:
    """
    Attempt up to 3 times (0, +5s, +5s) to fetch a 20 s MP4 from the camera.
    Uses a 30 second read timeout for each attempt.
    Returns the saved filepath on success, or None on permanent failure.
    """
    base_params = {
        "dw":        "sd",
        "filename":  segment_name,
        "starttime": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "endtime":   end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "index":     0,
        "sid":       0,
    }
    out_name = (
        f"{VIDEO_CLIPS_DIR}/clip_"
        f"{start_dt.strftime('%Y%m%d_%H%M%S')}_{end_dt.strftime('%H%M%S')}"
    )
    if unique_tag:
        out_name += f"_{unique_tag}"
    out_name += ".mp4"
    url = f"http://{camera_ip}/dataloader.cgi"

    max_retries = 2
    for attempt in range(max_retries + 1):
        params = base_params | {"uuid": str(uuid.uuid4())}
        try:
            logger.debug(
                f"Attempt {attempt+1}/{max_retries+1}: requesting clip {out_name} from {camera_ip}"
            )
            with requests.get(
                url,
                params=params,
                auth=HTTPDigestAuth(username, password),
                stream=True,
                # ``requests`` expects seconds; use a sensible 30 second timeout
                timeout=(10,30),
            ) as r:
                r.raise_for_status()
                with open(out_name, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            if not is_valid_mp4(out_name):
                os.remove(out_name)
                raise ValueError("Invalid clip downloaded")

            logger.debug(f"Successfully saved clip to {out_name}")
            return out_name

        except Exception as e:
            logger.error(f"Clip fetch attempt {attempt+1} failed: {e}", exc_info=True)
            if attempt < max_retries:
                time.sleep(5)
            else:
                logger.error(f"All {max_retries+1} attempts to fetch clip failed.")
                return None


def fetch_camera_frame(
    camera_ip: str,
    username: str,
    password: str,
    snapshot_path: str = "/snapshot.cgi",
    max_attempts: int = 3,
) -> bytes:
    """Return a JPEG snapshot from the camera via HTTP using Digest Auth.

    Tries the given ``snapshot_path`` (defaults to ``/snapshot.cgi``) and
    retries up to ``max_attempts`` times before raising an exception.
    """

    url = f"http://{camera_ip}{snapshot_path}"

    for attempt in range(max_attempts):
        try:
            # Use a moderate timeout so transient network issues do not block
            # the request for several minutes.
            r = requests.get(
                url,
                auth=HTTPDigestAuth(username, password),
                timeout=30,
            )
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            if not r.content or not content_type.startswith("image"):
                raise RuntimeError(
                    f"Unexpected response: status={r.status_code}, Content-Type={content_type}"
                )
            return r.content
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(0.2)
                continue
            else:
                logger.error("Failed fetching snapshot from %s", url)
                raise


def frame_from_video(path: str, offset_sec: float = 0.0) -> bytes:
    """Return a frame from a video file encoded as JPEG bytes.

    By default the first frame is returned.  If ``offset_sec`` is provided,
    the capture position is moved to that timestamp before reading.
    """

    if cv2 is None:
        raise RuntimeError("OpenCV not available")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video {path}")
    try:
        if offset_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, offset_sec * 1000)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from video")
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG")
        return buf.tobytes()
    finally:
        cap.release()


def fetch_exit_frame(
    camera_ip: str,
    username: str,
    password: str,
    
) -> bytes:
    """Return a JPEG snapshot from the camera around ``event_time``.

    The timestamp is currently unused and kept for backward compatibility.
    A snapshot is fetched directly from the camera using
    :func:`fetch_camera_frame` and returned as JPEG bytes.
    """

    frame_bytes = fetch_camera_frame(camera_ip, username, password)
    return frame_bytes


async def request_camera_clip_async(
    camera_ip: str,
    username: str,
    password: str,
    start_dt: datetime,
    end_dt: datetime,
    segment_name: str,
    unique_tag: str | None = None,
) -> Optional[str]:
    """Asynchronous wrapper around :func:`request_camera_clip`."""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        CAMERA_EXECUTOR,
        request_camera_clip,
        camera_ip,
        username,
        password,
        start_dt,
        end_dt,
        segment_name,
        unique_tag,
    )


async def fetch_camera_frame_async(
    camera_ip: str,
    username: str,
    password: str,
    snapshot_path: str = "/snapshot.cgi",
    max_attempts: int = 3,
) -> bytes:
    """Asynchronous wrapper around :func:`fetch_camera_frame`."""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        CAMERA_EXECUTOR,
        fetch_camera_frame,
        camera_ip,
        username,
        password,
        snapshot_path,
        max_attempts,
    )
