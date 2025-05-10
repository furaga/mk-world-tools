import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Any

from utils.logger import setup_logger

logger = setup_logger(__name__)


def imread_safe(
    filename: str, flags: int = cv2.IMREAD_COLOR, dtype: np.dtype = np.uint8
):
    """
    filenameに日本語が含まれても読み込めるようにしたcv2.imread
    """
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        logger.error(f"Failed to read image file: {filename} - {e}", exc_info=True)
        return None


def imwrite_safe(filename: str, img: np.ndarray, params: Optional[Any] = None):
    """
    filenameに日本語が含まれても書き込めるようにしたcv2.imwrite
    """
    try:
        ext = Path(filename).suffix
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode="w+b") as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Failed to write image file: {filename} - {e}", exc_info=True)
        return False
