import os
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from db import Base, engine, SessionLocal
from models import Location, Zone, Pole, Camera, Spot
from ocr_processor import exit_decision_from_frame

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

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
cam = Camera(pole_id=pole.id, api_code="C1", p_ip="1", portal_id=1)
session.add(cam)
session.commit()
spot = Spot(camera_id=cam.id, spot_number=1,
            p1_x=20, p1_y=20, p2_x=80, p2_y=20,
            p3_x=80, p3_y=80, p4_x=20, p4_y=80)
session.add(spot)
session.commit()
cam_id = cam.id
session.close()

def _dummy_model(boxes):
    class DummyModel:
        def __init__(self, boxes):
            self._boxes = boxes
        def __call__(self, arr, classes=None):
            class Box:
                def __init__(self, boxes):
                    self.xyxy = np.array(boxes)
                def __bool__(self):
                    return True
            class Res:
                def __init__(self, boxes):
                    self.boxes = Box(boxes)
            return [Res(self._boxes)]
    return DummyModel(boxes)

def test_exit_car_match():
    img = Image.new("RGB", (100, 100))
    boxes = [[30, 30, 60, 60]]
    with patch("ocr_processor.plate_model", _dummy_model(boxes)), \
         patch("ocr_processor.crop_and_save_car", return_value="p"), \
         patch("ocr_processor.same_entired_car", return_value=True):
        car, match = exit_decision_from_frame(img, cam_id, 1, 1)
    assert car is True
    assert match is True

def test_exit_car_mismatch():
    img = Image.new("RGB", (100, 100))
    boxes = [[30, 30, 60, 60]]
    with patch("ocr_processor.plate_model", _dummy_model(boxes)), \
         patch("ocr_processor.crop_and_save_car", return_value="p"), \
         patch("ocr_processor.same_entired_car", return_value=False):
        car, match = exit_decision_from_frame(img, cam_id, 1, 1)
    assert car is True
    assert match is False

def test_exit_no_car():
    img = Image.new("RGB", (100, 100))
    boxes = [[60, 60, 95, 95]]
    mock_same = MagicMock(return_value=True)
    with patch("ocr_processor.plate_model", _dummy_model(boxes)), \
         patch("ocr_processor.crop_and_save_car", return_value="p"), \
         patch("ocr_processor.same_entired_car", mock_same):
        car, match = exit_decision_from_frame(img, cam_id, 1, 1)
    assert car is False
    assert match is False
    assert mock_same.call_count == 0
