"""검사 결과를 CSV 리포트로 저장하는 모듈 (업무 자동화 산출물)."""
from pathlib import Path
from typing import Dict, List, Union
import csv


def write_csv_report(results: List[Dict], output_path: Union[str, Path]) -> None:
    """검사 결과 리스트를 CSV로 저장. 엑셀에서 한글이 깨지지 않도록 BOM 포함 인코딩 사용."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "verdict", "defect_probability"])
        for r in results:
            writer.writerow([
                r["file_name"],
                r["verdict"],
                f"{r['defect_probability']:.4f}",
            ])
