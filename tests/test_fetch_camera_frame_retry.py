from unittest.mock import MagicMock, patch
import requests
from requests.auth import HTTPDigestAuth

import camera_clip


def test_fetch_camera_frame_retries_until_frame():
    class DummyResp:
        def __init__(self, content=b"", exc=None):
            self.content = content
            self._exc = exc

        def raise_for_status(self):
            if self._exc:
                raise self._exc

    success = DummyResp(content=b"img")
    fail = requests.RequestException("boom")

    with patch('camera_clip.requests.get', side_effect=[fail, fail, success]) as req, \
         patch('camera_clip.time.sleep', return_value=None):
        result = camera_clip.fetch_camera_frame('ip', 'u', 'p', max_attempts=3)

    assert result == b"img"
    assert req.call_count == 3


def test_fetch_camera_frame_custom_path():
    resp = MagicMock()
    resp.content = b"x"
    resp.raise_for_status.return_value = None

    with patch('camera_clip.requests.get', return_value=resp) as req:
        camera_clip.fetch_camera_frame('ip', 'u', 'p', snapshot_path='/foo', max_attempts=1)

    req.assert_called_with(
        'http://ip/foo',
        auth=HTTPDigestAuth('u', 'p'),
        timeout=5
    )
