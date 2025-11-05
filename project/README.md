# 운수종사자 인지특성 기반 교통사고 위험 예측 대회 - 코드 제출용

## 개요
- 이 저장소는 Dacon 오프라인 평가 서버에서 실행 가능한 CPU 전용 파이프라인을 제공합니다.
- 목표: 인지·행동 검사 데이터(A/B)를 활용하여 사고 고위험군(Label=1) 확률을 예측.
- 평가식: 0.5(1-AUC) + 0.25(Brier) + 0.25(ECE)

## 폴더 구조
```
project/
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── feature_engineer.py
│   ├── model_utils.py
│   └── evaluate.py
├── train.py
├── script.py
├── requirements.txt
├── model/               # 학습 후 자동 생성
└── output/              # 추론 후 자동 생성
```

## 데이터 구조(예상)
```
data/
├── train.csv
├── test.csv
├── train/
│   ├── A.csv
│   └── B.csv
└── test/
    ├── A.csv
    └── B.csv
```

## 실행 방법
1) 의존성 설치
```bash
pip install -r project/requirements.txt
```

2) 학습(로컬 검증 포함)
```bash
python project/train.py
```
- 학습이 완료되면 `project/model/`에 `model.pkl`, `features.json`, `dtypes.json`이 저장됩니다.

3) 추론(제출 파일 생성)
```bash
python project/script.py
```
- 실행 후 `project/output/submission.csv`가 생성됩니다.

## 구현 노트
- 전처리: 수치형 중앙값 대치, 범주형 최빈값 대치 + 원-핫 인코딩(handle_unknown="ignore")
- 모델: LogisticRegression(max_iter=1000, class_weight='balanced') + CalibratedClassifierCV(method='sigmoid')
- 메모리 최적화: 숫자형/범주형 다운캐스팅
- 재현성: 시드 고정, 입력 피처 목록/타입 저장

## 주의사항
- 인터넷 접근 불가 환경을 가정하여 외부 다운로드는 사용하지 않습니다.
- 데이터 컬럼 구조가 일부 상이할 수 있어, 존재하지 않는 컬럼은 자동으로 무시되도록 구현되어 있습니다.
