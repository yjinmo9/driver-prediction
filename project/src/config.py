import os

# 한국어 주석: 프로젝트에서 공통으로 사용하는 경로, 시드, 상수 등을 정의하는 설정 파일

# 한국어 주석: 프로젝트 루트 경로를 기준으로 상대 경로를 안전하게 계산하기 위해 현재 파일 위치 사용
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# 한국어 주석: 데이터/모델/출력 디렉터리 경로 설정
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# 한국어 주석: 재현성을 위한 시드 고정 값
RANDOM_SEED = 42

# 한국어 주석: 병합/전처리 시 제거할 가능성이 높은 기본 컬럼들 (존재하지 않으면 무시)
DEFAULT_DROP_COLUMNS = [
    "Test_id",  # 한국어 주석: 모델 입력에서는 보통 식별자는 제거
    "TestDate",  # 한국어 주석: 대회에서 제공될 수 있는 날짜형 컬럼 가정 (없으면 무시)
]

# 한국어 주석: 모델 저장 파일명 및 메타데이터 파일명
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
FEATURES_FILE = os.path.join(MODEL_DIR, "features.json")
DTYPES_FILE = os.path.join(MODEL_DIR, "dtypes.json")

# 한국어 주석: A/B 분리 모델 저장 경로 (참고 코드 방식)
A_MODEL_FILE = os.path.join(MODEL_DIR, "model_A.pkl")
B_MODEL_FILE = os.path.join(MODEL_DIR, "model_B.pkl")
A_PREPROC_FILE = os.path.join(MODEL_DIR, "preproc_A.pkl")
B_PREPROC_FILE = os.path.join(MODEL_DIR, "preproc_B.pkl")
META_FILE = os.path.join(MODEL_DIR, "meta.json")

# 한국어 주석: 제출 파일 경로
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.csv")

# 한국어 주석: 학습/검증 분할 비율
VALID_SIZE = 0.2

# 한국어 주석: 캘리브레이션 교차검증 폴드 수 (CalibratedClassifierCV)
CALIBRATION_CV = 3  # 한국어 주석: 참고 코드는 3 사용

# 한국어 주석: 캘리브레이션 방법 (sigmoid vs isotonic)
CALIB_METHOD = "isotonic"  # 한국어 주석: 참고 코드는 isotonic 사용

# 한국어 주석: 앙상블을 위한 여러 시드 값 (참고 코드 방식)
ENSEMBLE_SEEDS = (42, 202, 777)

# 한국어 주석: HistGradientBoostingClassifier 기본 파라미터 (참고 코드 방식)
BASE_HGB_PARAMS = {
    "learning_rate": 0.06,  # 한국어 주석: 학습률
    "max_iter": 300,  # 한국어 주석: 최대 반복 횟수
    "max_depth": None,  # 한국어 주석: 최대 깊이 (None = 제한 없음)
    "max_leaf_nodes": 63,  # 한국어 주석: 최대 잎 노드 수
    "min_samples_leaf": 20,  # 한국어 주석: 잎 노드 최소 샘플 수
    "l2_regularization": 0.0,  # 한국어 주석: L2 정규화
    "early_stopping": True,  # 한국어 주석: 조기 종료 활성화
    "validation_fraction": 0.12,  # 한국어 주석: 검증용 데이터 비율
    "n_iter_no_change": 25,  # 한국어 주석: 변화 없으면 종료
    "class_weight": "balanced",  # 한국어 주석: 불균형 데이터 가중치
}


