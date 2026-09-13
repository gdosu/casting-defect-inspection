# 주조품 표면 결함 검사 AI

주조(casting) 제품 이미지를 보고 정상(OK)/결함(DEFECT) 여부를 자동으로 판정하는 간단한 비전검사 파이프라인입니다.
제조 현장 비전검사 AI 시스템 개발, 업무 자동화 파이프라인 구축 역량을 보여주기 위한 포트폴리오용 프로젝트입니다.

## 데이터셋

[Kaggle - Casting Product Image Data for Quality Inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
(펌프 임펠러 주조품의 정상/결함 이미지, `ok_front` / `def_front` 로 라벨링, train/test 스플릿 제공)

용량 문제로 데이터셋 원본은 저장소에 포함하지 않았습니다. 직접 학습을 재현하려면 위 링크에서 다운로드한 뒤
`dataset/casting_data/casting_data/{train,test}/{ok_front,def_front}` 경로에 압축을 풀어주세요.
(`sample_images/`에 데모용 샘플 몇 장만 별도로 포함해두었습니다.)

## 접근 방법

1. 이미지를 그레이스케일 64x64로 전처리 ([src/preprocess.py](src/preprocess.py))
2. HOG(Histogram of Oriented Gradients) 특징 추출 ([src/features.py](src/features.py))
3. 로지스틱 회귀로 OK/DEFECT 이진 분류 학습 ([src/model.py](src/model.py))

딥러닝 없이도 간단한 특징 추출 + 고전적 분류기 조합으로 실용적인 성능을 낼 수 있다는 것을 보여주는 데 초점을 맞췄습니다.

## 결과

held-out 테스트셋(715장) 기준:

| 지표 | 값 |
|---|---|
| Accuracy | 0.9469 |
| OK precision / recall | 0.90 / 0.96 |
| DEFECT precision / recall | 0.98 / 0.94 |

자세한 리포트: [assets/metrics.txt](assets/metrics.txt)

## 실행 방법

```powershell
pip install -r requirements.txt

# 1) 모델 학습 (dataset/ 폴더에 데이터셋이 준비되어 있어야 함)
python train.py

# 2) 폴더 단위 배치 검사 + CSV 리포트 생성 (자동화 파이프라인)
python run_batch.py --input sample_images --output output

# 3) 이미지 업로드 후 즉시 판정해보는 웹 데모
streamlit run app.py
```

학습된 모델([assets/model.joblib](assets/model.joblib))은 저장소에 이미 포함되어 있어서,
데이터셋 없이 `run_batch.py` / `app.py` 로 바로 데모를 실행해볼 수 있습니다.

## 프로젝트 구조

```
├─ src/
│  ├─ preprocess.py   # 이미지 로딩/전처리
│  ├─ features.py     # HOG 특징 추출
│  ├─ model.py         # 학습/평가/저장/추론
│  └─ report.py        # 검사 결과 CSV 리포트 생성
├─ train.py            # 모델 학습 스크립트
├─ run_batch.py         # 폴더 단위 배치 검사 스크립트
├─ app.py               # Streamlit 웹 데모
├─ assets/              # 학습된 모델, 평가 리포트
└─ sample_images/       # 데모용 샘플 이미지
```

## 한계 및 향후 개선

- 이미지 해상도를 64x64로 축소해 처리하기 때문에 미세한 결함은 놓칠 수 있음
- HOG는 전역적인 형태/질감 특징 위주라, 결함의 정확한 위치를 짚어주지는 못함 (판정만 가능)
- CNN 기반 모델이나 결함 영역을 직접 표시하는 세그멘테이션 모델로 확장하면 정확도와 설명력을 더 높일 수 있음
