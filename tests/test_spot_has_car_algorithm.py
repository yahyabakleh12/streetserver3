import os
from unittest.mock import patch
from PIL import Image
import numpy as np

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from db import Base, engine, SessionLocal
from models import Location, Zone, Pole, Camera, Spot
from ocr_processor import spot_has_car

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

class DummyModel:
    def __init__(self, boxes):
        self._boxes = boxes
    def __call__(self, arr):
        class Box:
            def __init__(self, boxes):
                self.xyxy = np.array(boxes)
            def __bool__(self):
                return True
        class Res:
            def __init__(self, boxes):
                self.boxes = Box(boxes)
        return [Res(self._boxes)]

def test_car_inside_polygon():
    img = Image.new("RGB", (100, 100))
    boxes = [[30, 30, 60, 60]]
    with patch("ocr_processor.plate_model", DummyModel(boxes)):
        assert spot_has_car(img, cam_id, 1)

def test_car_below_threshold():
    img = Image.new("RGB", (100, 100))
    boxes = [[60, 60, 95, 95]]
    with patch("ocr_processor.plate_model", DummyModel(boxes)):
        assert not spot_has_car(img, cam_id, 1)

