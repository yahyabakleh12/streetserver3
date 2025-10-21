import os

TEST_DB = "sqlite:///./test.db"
os.environ["DATABASE_URL"] = TEST_DB

from network import parse_ping_status


def test_parse_online():
    output = (
        "Pinging 192.168.169.88 with 32 bytes of data:\n"
        "Reply from 192.168.169.88: bytes=32 time=26ms TTL=62\n"
        "Reply from 192.168.169.88: bytes=32 time=45ms TTL=62\n"
    )
    assert parse_ping_status(output) == "ONLINE"


def test_parse_request_timeout():
    output = (
        "Pinging 192.168.169.51 with 32 bytes of data:\n"
        "Request timed out.\n"
    )
    assert parse_ping_status(output) == "OFFLINE"


def test_parse_unreachable():
    output = (
        "Reply from 192.168.169.1: Destination host unreachable.\n"
    )
    assert parse_ping_status(output) == "OFFLINE"
