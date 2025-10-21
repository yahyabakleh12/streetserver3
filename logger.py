# logger.py

import logging
import sys

# Set up a root logger that writes to both console and a file.
LOG_FORMAT = (
    "%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] "
    "- %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("parking_app.log", encoding="utf-8")
    ]
)

# “parking_app” logger
logger = logging.getLogger("parking_app")
