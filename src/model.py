"""결함 분류 모델(학습/평가/저장/추론)을 담당하는 모듈.

라벨: 0 = OK(정상), 1 = DEFECT(결함)
"""
from pathlib import Path
from typing import Tuple, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.features import extract_hog
from src.preprocess import load_grayscale

LABELS = {0: "OK", 1: "DEFECT"}


def load_split(base_dir: Union[str, Path], limit_per_class: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """base_dir/ok_front, base_dir/def_front 폴더에서 이미지를 읽어 (특징 배열, 라벨 배열)로 변환."""
    base_dir = Path(base_dir)
    features, labels = [], []
    for label, folder_name in [(0, "ok_front"), (1, "def_front")]:
        files = sorted((base_dir / folder_name).glob("*.jpeg"))
        if limit_per_class is not None:
            files = files[:limit_per_class]
        for f in files:
            image = load_grayscale(f)
            features.append(extract_hog(image))
            labels.append(label)
    return np.array(features), np.array(labels)


def train(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """HOG 특징을 입력으로 받는 로지스틱 회귀 분류기를 학습."""
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """정확도와 상세 분류 리포트를 계산."""
    pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "report": classification_report(y_test, pred, target_names=["OK", "DEFECT"]),
    }


def save_model(clf: LogisticRegression, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)


def load_model(path: Union[str, Path]) -> LogisticRegression:
    return joblib.load(path)


def predict(clf: LogisticRegression, image: np.ndarray) -> Tuple[str, float]:
    """이미지 한 장에 대해 (판정 라벨, 결함일 확률)을 반환."""
    feature = extract_hog(image).reshape(1, -1)
    proba = clf.predict_proba(feature)[0]
    defect_proba = proba[1]
    label = LABELS[int(defect_proba >= 0.5)]
    return label, float(defect_proba)
