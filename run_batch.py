"""이미지 폴더 전체에 학습된 모델을 적용해 CSV 리포트를 자동 생성하는 배치 스크립트.

사용 예:
    python run_batch.py --input sample_images --output output
"""
import argparse
from pathlib import Path

from src.model import load_model, predict
from src.preprocess import load_grayscale
from src.report import write_csv_report

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets" / "model.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="비전검사 배치 파이프라인")
    parser.add_argument("--input", default="sample_images", help="검사할 이미지가 담긴 폴더")
    parser.add_argument("--output", default="output", help="결과 리포트를 저장할 폴더")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    clf = load_model(MODEL_PATH)
    image_paths = sorted(input_dir.glob("*.jpeg"))
    if not image_paths:
        print(f"검사할 이미지가 없습니다: {input_dir}")
        return

    results = []
    for path in image_paths:
        image = load_grayscale(path)
        verdict, defect_probability = predict(clf, image)
        results.append({
            "file_name": path.name,
            "verdict": verdict,
            "defect_probability": defect_probability,
        })
        print(f"{path.name}: {verdict} (defect_probability={defect_probability:.4f})")

    write_csv_report(results, output_dir / "report.csv")
    print(f"\n리포트 저장 완료: {output_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
