"""이미지 한 장을 업로드하면 즉시 결함 여부를 판정해주는 Streamlit 데모 앱.

실행: streamlit run app.py
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.model import load_model, predict

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets" / "model.joblib"


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def read_uploaded_image(uploaded_file, size=(64, 64)) -> np.ndarray:
    """업로드된 파일을 그레이스케일 배열로 변환 (디스크에 저장하지 않고 메모리에서 바로 처리)."""
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, size)


st.set_page_config(page_title="주조품 비전검사 데모", layout="centered")
st.title("주조품 표면 결함 검사 데모")
st.caption("HOG 특징 + 로지스틱 회귀 분류기로 학습된 모델을 이용해 정상/결함 여부를 판정합니다.")

uploaded_file = st.file_uploader("검사할 이미지를 업로드하세요 (jpeg)", type=["jpeg", "jpg"])

if uploaded_file is not None:
    clf = get_model()
    image = read_uploaded_image(uploaded_file)
    verdict, defect_probability = predict(clf, image)

    st.image(image, caption="입력 이미지", channels="GRAY", width=300)
    st.metric("판정 결과", verdict)
    st.write(f"결함 확률: {defect_probability:.2%}")
