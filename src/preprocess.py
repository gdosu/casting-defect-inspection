"""이미지 로딩을 담당하는 모듈."""
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np


def load_grayscale(path: Union[str, Path], size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """이미지를 그레이스케일로 읽고 지정한 크기로 통일해서 반환."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return cv2.resize(image, size)
