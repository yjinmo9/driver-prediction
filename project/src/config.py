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
PREPROC_ALL_FILE = os.path.join(MODEL_DIR, "preproc_all.pkl")

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
    "learning_rate": 0.035,
    "max_iter": 1100,
    "max_depth": None,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 60,
    "l2_regularization": 0.7,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 50,
    "class_weight": "balanced",
}

# 한국어 주석: 혼합 앙상블 사용 설정
ENABLE_LGBM = True
ENABLE_XGB = True

# 한국어 주석: LightGBM 기본 파라미터 (혼합 앙상블용)
LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "n_estimators": 800,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "lambda_l2": 0.3,
    "n_jobs": -1,
}

# 한국어 주석: XGBoost 기본 파라미터 (혼합 앙상블용)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "learning_rate": 0.05,
    "n_estimators": 800,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "eval_metric": "logloss",
    "n_jobs": -1,
    # "tree_method": "hist",  # GPU/환경에 맞게 필요시 활성화
}

# 한국어 주석: LightGBM 기본 파라미터
BASE_LIGHTGBM_PARAMS = {
    "learning_rate": 0.06,
    "n_estimators": 100,
    "max_depth": -1,  # 제한 없음
    "num_leaves": 63,
    "min_child_samples": 20,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "early_stopping_rounds": 25,
    "class_weight": "balanced",
    "verbose": -1,  # 출력 억제
}

# 한국어 주석: CatBoost 기본 파라미터
BASE_CATBOOST_PARAMS = {
    "learning_rate": 0.06,
    "iterations": 100,
    "depth": 6,
    "l2_leaf_reg": 0.0,
    "early_stopping_rounds": 25,
    "class_weights": "balanced",
    "verbose": False,  # 출력 억제
    "random_seed": RANDOM_SEED,
}

# 한국어 주석: XGBoost 기본 파라미터
BASE_XGBOOST_PARAMS = {
    "learning_rate": 0.06,
    "n_estimators": 100,
    "max_depth": 6,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "early_stopping_rounds": 25,
    "scale_pos_weight": 1,  # balanced로 자동 계산
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
}


