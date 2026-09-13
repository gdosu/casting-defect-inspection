"""라벨링된 train/test 데이터로 결함 분류 모델을 학습하고 성능을 평가하는 스크립트.

사용 예:
    python train.py
"""
from pathlib import Path

from src.model import train, evaluate, save_model, load_split

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "dataset" / "casting_data" / "casting_data" / "train"
TEST_DIR = ROOT / "dataset" / "casting_data" / "casting_data" / "test"
MODEL_PATH = ROOT / "assets" / "model.joblib"


def main() -> None:
    print("학습 데이터 로딩 중...")
    X_train, y_train = load_split(TRAIN_DIR)
    print(f"학습 이미지 수: {len(y_train)}")

    print("테스트 데이터 로딩 중...")
    X_test, y_test = load_split(TEST_DIR)
    print(f"테스트 이미지 수: {len(y_test)}")

    print("모델 학습 중...")
    clf = train(X_train, y_train)

    metrics = evaluate(clf, X_test, y_test)
    print(f"\n테스트 정확도: {metrics['accuracy']:.4f}\n")
    print(metrics["report"])

    save_model(clf, MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")

    metrics_path = ROOT / "assets" / "metrics.txt"
    metrics_path.write_text(
        f"test accuracy: {metrics['accuracy']:.4f}\n\n{metrics['report']}", encoding="utf-8"
    )
    print(f"평가 리포트 저장 완료: {metrics_path}")


if __name__ == "__main__":
    main()
