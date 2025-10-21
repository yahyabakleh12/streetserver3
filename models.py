# models.py

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Enum,
    DateTime,
    JSON,
    ForeignKey,
    Table,
    Text,
)
from sqlalchemy.orm import relationship
from db import Base

# Association tables for many-to-many relationships
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", Integer, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
)

role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", Integer, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    Column(
        "permission_id",
        Integer,
        ForeignKey("permissions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class User(Base):
    """Application user for authentication."""

    __tablename__ = "users"

    id             = Column(Integer, primary_key=True, index=True)
    username       = Column(String(50), unique=True, nullable=False)
    hashed_password= Column(String(128), nullable=False)
    created_at     = Column(DateTime, default=datetime.utcnow)

    roles = relationship(
        "Role",
        secondary=user_roles,
        back_populates="users",
    )


class Role(Base):
    __tablename__ = "roles"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(50), unique=True, nullable=False)
    description = Column(String(255), nullable=True)

    users = relationship(
        "User",
        secondary=user_roles,
        back_populates="roles",
    )
    permissions = relationship(
        "Permission",
        secondary=role_permissions,
        back_populates="roles",
    )


class Permission(Base):
    __tablename__ = "permissions"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(50), unique=True, nullable=False)
    description = Column(String(255), nullable=True)

    roles = relationship(
        "Role",
        secondary=role_permissions,
        back_populates="permissions",
    )


class Location(Base):
    __tablename__ = "locations"
    id             = Column(Integer, primary_key=True, index=True)
    name           = Column(String(100), nullable=False)
    code           = Column(String(50),  nullable=False, unique=True)
    portal_name    = Column(String(100), nullable=False)
    portal_password= Column(String(100), nullable=False)
    ip_schema      = Column(String(100), nullable=False)
    parkonic_api_token = Column(String(255), nullable=True)
    camera_user       = Column(String(100), nullable=True)
    camera_pass       = Column(String(100), nullable=True)
    parameters     = Column(JSON, nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow)

    poles  = relationship("Pole", back_populates="location")
    zones  = relationship("Zone", back_populates="location")


class Zone(Base):
    __tablename__ = "zones"
    id          = Column(Integer, primary_key=True, index=True)
    code        = Column(String(50), nullable=False)
    parameters  = Column(JSON, nullable=True)
    location_id = Column(Integer, ForeignKey("locations.id", ondelete="CASCADE"), nullable=False)

    location = relationship("Location", back_populates="zones")
    poles    = relationship("Pole", back_populates="zone")


class Pole(Base):
    __tablename__ = "poles"
    id                = Column(Integer, primary_key=True, index=True)
    zone_id           = Column(Integer, ForeignKey("zones.id", ondelete="CASCADE"), nullable=False)
    name              = Column(String(100), nullable=True)
    code              = Column(String(50), nullable=False)
    location_id       = Column(Integer, ForeignKey("locations.id", ondelete="CASCADE"), nullable=False)
    number_of_cameras = Column(Integer, default=0)
    server            = Column(String(100), nullable=True)
    router            = Column(String(100), nullable=True)
    router_ip         = Column(String(45), nullable=True)
    router_vpn_ip     = Column(String(45), nullable=True)
    location_coordinates = Column(String(255), nullable=True)
    api_pole_id       = Column(Integer, nullable=True)

    cameras  = relationship("Camera", back_populates="pole")
    location = relationship("Location", back_populates="poles")
    zone     = relationship("Zone", back_populates="poles")


class Camera(Base):
    __tablename__ = "cameras"
    id                = Column(Integer, primary_key=True, index=True)
    pole_id           = Column(Integer, ForeignKey("poles.id", ondelete="CASCADE"), nullable=False)
    name              = Column(String(100), nullable=True)
    api_code          = Column(String(100), nullable=False)
    p_ip              = Column(String(45),  nullable=False)
    number_of_parking = Column(Integer, default=0)
    vpn_ip            = Column(String(45),  nullable=True)
    portal_id        = Column(Integer, nullable=True)
    status            = Column(
        Enum("ONLINE", "OFFLINE", name="camera_status"),
        nullable=False,
        default="OFFLINE",
    )

    # Relationships (optional)
    reports          = relationship("Report", back_populates="camera")
    plate_logs       = relationship("PlateLog", back_populates="camera")
    tickets          = relationship("Ticket", back_populates="camera")
    manual_reviews   = relationship("ManualReview", back_populates="camera")
    pole             = relationship("Pole", back_populates="cameras")
    clip_requests    = relationship("ClipRequest", back_populates="camera")
    spots            = relationship("Spot", back_populates="camera")
    crop_zones       = relationship("CropZone", back_populates="camera")


class Spot(Base):
    __tablename__ = "spots"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    spot_number = Column(Integer, nullable=False)
    p1_x = Column(Integer, nullable=False)
    p1_y = Column(Integer, nullable=False)
    p2_x = Column(Integer, nullable=False)
    p2_y = Column(Integer, nullable=False)
    p3_x = Column(Integer, nullable=False)
    p3_y = Column(Integer, nullable=False)
    p4_x = Column(Integer, nullable=False)
    p4_y = Column(Integer, nullable=False)
    status = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    camera = relationship("Camera", back_populates="spots")

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        xs = [self.p1_x, self.p2_x, self.p3_x, self.p4_x]
        ys = [self.p1_y, self.p2_y, self.p3_y, self.p4_y]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def bbox_x1(self) -> int:
        return self.bbox[0]

    @property
    def bbox_y1(self) -> int:
        return self.bbox[1]

    @property
    def bbox_x2(self) -> int:
        return self.bbox[2]

    @property
    def bbox_y2(self) -> int:
        return self.bbox[3]


class CropZone(Base):
    __tablename__ = "crop_zones"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    points = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    camera = relationship("Camera", back_populates="crop_zones")


class Report(Base):
    __tablename__ = "reports"
    id         = Column(Integer, primary_key=True, index=True)
    camera_id  = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    event       = Column(String(100),   nullable=False)
    report_type = Column(String(50),    nullable=False)
    timestamp   = Column(DateTime,      nullable=False)
    payload     = Column(JSON,          nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)

    camera    = relationship("Camera", back_populates="reports")


class PlateLog(Base):
    __tablename__ = "plate_logs"
    id           = Column(Integer, primary_key=True, index=True)
    camera_id    = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    car_id       = Column(String(50), nullable=True)
    plate_number = Column(String(20), nullable=True)
    plate_code   = Column(String(10), nullable=True)
    plate_city   = Column(String(50), nullable=True)
    confidence   = Column(Integer, nullable=True)
    image_path   = Column(String(255), nullable=False)
    status       = Column(Enum("READ", "UNREAD", name="plate_status"), nullable=False, default="UNREAD")
    attempt_ts   = Column(DateTime, default=datetime.utcnow)

    camera    = relationship("Camera", back_populates="plate_logs")


class Ticket(Base):
    __tablename__ = "tickets"
    id               = Column(Integer, primary_key=True, index=True)
    camera_id        = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    spot_number      = Column(Integer,    nullable=False)
    plate_number     = Column(String(20), nullable=False)
    plate_code       = Column(String(10), nullable=True)
    plate_city       = Column(String(50), nullable=True)
    confidence       = Column(Integer,    nullable=True)
    entry_time       = Column(DateTime,   nullable=False)
    exit_time        = Column(DateTime,   nullable=True)
    parkonic_trip_id = Column(Integer,    nullable=True)
    image_base64    = Column(Text, nullable=True)
    entry_image_path = Column(String(255), nullable=True)
    exit_clip_path   = Column(String(255), nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)

    camera    = relationship("Camera", back_populates="tickets")
    pending_payloads = relationship(
        "PendingTicketPayload",
        back_populates="ticket",
        cascade="all, delete-orphan",
    )


class ManualReview(Base):
    __tablename__ = "manual_reviews"
    id             = Column(Integer, primary_key=True, index=True)
    camera_id      = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    spot_number    = Column(Integer,    nullable=False)
    event_time     = Column(DateTime,   nullable=False)
    image_path     = Column(String(255), nullable=False)
    clip_path      = Column(String(255), nullable=True)
    ticket_id      = Column(Integer, ForeignKey("tickets.id", ondelete="SET NULL"), nullable=True)
    plate_status   = Column(Enum("READ", "UNREAD", name="manual_plate_status"), nullable=False)
    plate_image    = Column(String(255), nullable=False)
    snapshot_folder = Column(String(255), nullable=False)
    review_status  = Column(Enum("PENDING", "RESOLVED", name="review_status"), nullable=False, default="PENDING")
    created_at     = Column(DateTime, default=datetime.utcnow)

    camera    = relationship("Camera", back_populates="manual_reviews")
    ticket    = relationship("Ticket")


class ClipRequest(Base):
    __tablename__ = "clip_requests"

    id         = Column(Integer, primary_key=True, index=True)
    camera_id  = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time   = Column(DateTime, nullable=False)
    status     = Column(Enum("PENDING", "COMPLETED", "FAILED", name="clip_request_status"), nullable=False, default="PENDING")
    clip_path  = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    camera = relationship("Camera", back_populates="clip_requests")


class QueuedRequest(Base):
    """Stores the path to a raw request that is pending analysis."""

    __tablename__ = "queued_requests"

    id         = Column(Integer, primary_key=True, index=True)
    file_path  = Column(String(255), nullable=False)
    ts         = Column(String(20),  nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)



class PendingTicketPayload(Base):
    """Durable store for ticket payloads awaiting delivery."""

    __tablename__ = "pending_ticket_payloads"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(
        Integer,
        ForeignKey("tickets.id", ondelete="CASCADE"),
        nullable=True,
    )
    payload = Column(JSON, nullable=False)
    attempt_count = Column(Integer, nullable=False, default=0)
    last_attempt_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    ticket = relationship("Ticket", back_populates="pending_payloads")


