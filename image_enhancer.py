import os
from logger import logger

try:
    import cv2
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    _AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    RRDBNet = None  # type: ignore
    RealESRGANer = None  # type: ignore
    torch = None  # type: ignore
    cv2 = None  # type: ignore
    _AVAILABLE = False


_upsampler = None

def _init_model():
    global _upsampler
    if not _AVAILABLE:
        return None
    if _upsampler is None:
        model_path = os.environ.get("REAL_ESRGAN_MODEL_PATH", "weights/RealESRGAN_x4plus.pth")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                         num_block=23, num_grow_ch=32, scale=4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device,
        )
    return _upsampler


def enhance_image_array(img_bgr):
    """Return an enhanced BGR image array or the original if enhancement fails."""
    upsampler = _init_model()
    if upsampler is None:
        return img_bgr
    try:
        output, _ = upsampler.enhance(img_bgr, outscale=4)
        return output
    except Exception:
        logger.error("RealESRGAN enhancement failed", exc_info=True)
        return img_bgr
