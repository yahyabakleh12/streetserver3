# network.py

import time
import asyncio
import requests
import platform
import atexit
from datetime import datetime
from subprocess import DEVNULL, run

import httpx

from logger import logger
from db import SessionLocal
from models import Camera


# Reuse a single session for all outbound HTTP requests
session = requests.Session()
atexit.register(session.close)


def parse_ping_status(output: str) -> str:
    """Return "ONLINE" if ping output indicates success, else "OFFLINE"."""
    out = output.lower()
    if any(word in out for word in [
        "destination host unreachable",
        "request timed out",
        "network is unreachable",
        "100% packet loss",
    ]):
        return "OFFLINE"
    if "reply from" in out or "bytes from" in out or "ttl=" in out:
        return "ONLINE"
    return "OFFLINE"

def send_request_with_retry(
    url: str,
    payload: dict,
    *,
    timeout: float = 10.0,
    max_retries: int = 2,
    backoff: float = 1.0,
) -> str:
    """
    POST ``payload`` to ``url`` with optional retries and exponential backoff.

    The function simply returns ``requests.Response.text``.  Callers are
    responsible for parsing the returned value (some endpoints return a JSON
    string while others may double encode the JSON).

    Raises any ``requests`` exception after the final retry.
    """
    for attempt in range(max_retries + 1):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.text  # some endpoints return a quoted JSON-string
        except Exception as e:
            logger.error(
                "send_request_with_retry attempt %d failed: %s",
                attempt + 1,
                e,
                exc_info=True,
            )
            if attempt < max_retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise


async def send_request_with_retry_async(
    url: str,
    payload: dict,
    *,
    timeout: float = 10.0,
    max_retries: int = 2,
    backoff: float = 1.0,
) -> str:
    """
    Asynchronous counterpart to :func:`send_request_with_retry` using
    ``httpx.AsyncClient``.

    The payload is POSTed to the URL with optional retries and exponential
    backoff. The response body is returned as text.
    """
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.text
        except Exception as e:
            logger.error(
                "send_request_with_retry_async attempt %d failed: %s",
                attempt + 1,
                e,
                exc_info=True,
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff * (2 ** attempt))
            else:
                raise


def ping_all_cameras(log_path: str = "cameras.log", interval: int = 120) -> None:
    """Continuously ping all cameras and update their online status.

    Every ``interval`` seconds each camera's IP is pinged once with a 1 second
    timeout. A line of the form ``"<timestamp> camera <ip> <status>"`` is
    appended to ``log_path`` for every camera. The camera's ``status`` field is
    updated in the database if it has changed.
    """

    while True:
        db = SessionLocal()
        try:
            cameras = db.query(Camera).all()
            with open(log_path, "a", encoding="utf-8") as log:
                for cam in cameras:
                    ping_cmd = ["ping"]
                    if platform.system().lower().startswith("win"):
                        ping_cmd += ["-n", "1", "-w", "1000"]
                    else:
                        ping_cmd += ["-c", "1", "-W", "1000"]
                    ping_cmd.append(cam.p_ip)
                    result = run(ping_cmd, capture_output=True, text=True)
                    #  print(result)
                    status = parse_ping_status(result.stdout + result.stderr)
                    log.write(
                        f"{datetime.utcnow().isoformat()} camera {cam.p_ip} {status}\n"
                    )
                    if cam.status != status:
                        cam.status = status
            db.commit()
        except Exception as e:
            logger.error("Failed pinging cameras: %s", e, exc_info=True)
        finally:
            db.close()
        time.sleep(interval)
