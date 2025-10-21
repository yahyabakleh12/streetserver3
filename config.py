# config.py

# ─────────────────────────────────────────────────────────────────────────────
# Database connection string provided via the `DATABASE_URL` environment
# variable. No fallback credentials are stored in the repository.
# ─────────────────────────────────────────────────────────────────────────────
import os

# The application requires a database connection string in `DATABASE_URL`.
# No default credentials are provided.
DATABASE_URL = os.environ.get("DATABASE_URL")

# ─────────────────────────────────────────────────────────────────────────────
# API Tokens
# ─────────────────────────────────────────────────────────────────────────────
# Token for the OCR service must be provided via the `OCR_TOKEN` environment
# variable.
OCR_TOKEN = os.environ.get("OCR_TOKEN")
# Location specific configuration now stores the Parkonic API token and camera
# credentials.  The global constants previously defined here have been
# deprecated.

# ─────────────────────────────────────────────────────────────────────────────
# YOLO model path (on CPU)
# ─────────────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = "models/yolo11x.pt"
OS_NET_MODEL = "models/osnet_car_final.pth"
VEHICLE_CLASSES = [2, 5, 7]
# RealESRGAN model weights path
REAL_ESRGAN_MODEL_PATH = os.environ.get(
    "REAL_ESRGAN_MODEL_PATH",
    "weights/RealESRGAN_x4plus.pth",
)

API_LOCATION_ID = 213

# Base URL of this server used when constructing download links in
# outgoing API payloads. Defaults to the local development URL if not
# provided via the `SERVER_BASE` environment variable.
SERVER_BASE = os.environ.get("SERVER_BASE", "http://localhost:8000")
