"""이미지에서 HOG(Histogram of Oriented Gradients) 특징을 추출하는 모듈.

딥러닝 없이도 표면의 형태/질감 패턴을 수치화할 수 있어 간단한 분류기와 궁합이 좋다.
"""
import numpy as np
from skimage.feature import hog


def extract_hog(image: np.ndarray) -> np.ndarray:
    """그레이스케일 이미지 한 장을 HOG 특징 벡터로 변환."""
    return hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
