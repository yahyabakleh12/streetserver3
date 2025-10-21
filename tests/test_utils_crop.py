import os
import numpy as np
from PIL import Image

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from db import Base, engine, SessionLocal
from models import Location, Zone, Pole, Camera, Spot
from utils import is_same_image

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
spot = Spot(
    camera_id=cam.id,
    spot_number=1,
    p1_x=10,
    p1_y=10,
    p2_x=20,
    p2_y=10,
    p3_x=20,
    p3_y=20,
    p4_x=10,
    p4_y=20,
)
session.add(spot)
session.commit()
cam_id = cam.id
session.close()

def test_is_same_image_crops(tmp_path):
    img1 = np.zeros((30,30), dtype=np.uint8)
    img1[10:20,10:20] = 255
    img1[0:5,0:5] = 50
    img2 = np.zeros((30,30), dtype=np.uint8)
    img2[10:20,10:20] = 255
    img2[0:5,0:5] = 200
    p1 = tmp_path / "i1.jpg"
    p2 = tmp_path / "i2.jpg"
    Image.fromarray(img1).save(p1)
    Image.fromarray(img2).save(p2)

    assert is_same_image(str(p1), str(p2), camera_id=cam_id, spot_number=1)


def teardown_module(module):
    Base.metadata.drop_all(bind=engine)
    try:
        os.remove("test.db")
    except FileNotFoundError:
        pass
