import json
import os
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import __version__ as sklver

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from .config import (
    A_MODEL_FILE,
    B_MODEL_FILE,
    A_PREPROC_FILE,
    B_PREPROC_FILE,
    META_FILE,
    RANDOM_SEED,
    BASE_HGB_PARAMS,
    BASE_LIGHTGBM_PARAMS,
    BASE_CATBOOST_PARAMS,
    BASE_XGBOOST_PARAMS,
    CALIB_METHOD,
    CALIBRATION_CV,
    MODEL_FILE,
    PREPROC_ALL_FILE,
)


# 한국어 주석: HistGradientBoostingClassifier 생성 (참고 코드 방식)
def create_base_estimator(seed: int = 42) -> HistGradientBoostingClassifier:
    # 한국어 주석: config에서 정의한 HGB 파라미터를 사용하여 모델 생성
    params = BASE_HGB_PARAMS.copy()
    params["random_state"] = seed
    return HistGradientBoostingClassifier(**params)


# 한국어 주석: CalibratedClassifierCV의 파라미터명이 버전에 따라 다를 수 있어 안전하게 처리
def create_calibrated_model(estimator) -> CalibratedClassifierCV:
    # 한국어 주석: sklearn 버전에 따라 파라미터명이 달라서 try-except로 처리
    try:
        major, minor, *_ = map(int, sklver.split(".")[:2])
    except Exception:
        major, minor = 1, 4
    
    kw = dict(method=CALIB_METHOD, cv=CALIBRATION_CV)
    if (major, minor) >= (1, 4):
        # 한국어 주석: 최신 버전 (sklearn>=1.4)
        return CalibratedClassifierCV(estimator=estimator, **kw)
    else:
        # 한국어 주석: 구버전 호환 (sklearn<=1.3)
        return CalibratedClassifierCV(base_estimator=estimator, **kw)


# 한국어 주석: 3개의 다른 시드로 학습한 모델의 확률을 평균내는 앙상블 클래스
class AvgProbaEnsemble:
    def __init__(self, models: List):
        # 한국어 주석: 여러 개의 캘리브레이션된 모델을 저장
        self.models = models

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # 한국어 주석: 각 모델의 예측 확률을 구한 후 평균
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)


# 한국어 주석: LightGBM 모델 생성
def create_lightgbm_model(seed: int = RANDOM_SEED) -> lgb.LGBMClassifier:
    params = BASE_LIGHTGBM_PARAMS.copy()
    params["random_state"] = seed
    params["objective"] = "binary"
    params["metric"] = "binary_logloss"
    # Calibration과 호환되도록 early_stopping_rounds는 None으로 설정 (fit에서 직접 처리)
    params.pop("early_stopping_rounds", None)
    return lgb.LGBMClassifier(**params)


# 한국어 주석: CatBoost 모델 생성
def create_catboost_model(seed: int = RANDOM_SEED, class_weights: dict = None) -> cb.CatBoostClassifier:
    params = BASE_CATBOOST_PARAMS.copy()
    params["random_seed"] = seed
    params["loss_function"] = "Logloss"
    params["task_type"] = "CPU"
    # Calibration과 호환되도록 early_stopping_rounds는 fit에서 직접 처리
    params.pop("early_stopping_rounds", None)
    # class_weights가 문자열이면 None으로 설정 (나중에 fit에서 계산)
    if "class_weights" in params and params["class_weights"] == "balanced":
        params.pop("class_weights", None)
    if class_weights is not None:
        params["class_weights"] = class_weights
    return cb.CatBoostClassifier(**params)


# 한국어 주석: XGBoost 모델 생성
def create_xgboost_model(seed: int = RANDOM_SEED) -> xgb.XGBClassifier:
    params = BASE_XGBOOST_PARAMS.copy()
    params["random_state"] = seed
    params["objective"] = "binary:logistic"
    # Calibration과 호환되도록 early_stopping_rounds는 fit에서 직접 처리
    params.pop("early_stopping_rounds", None)
    return xgb.XGBClassifier(**params)


# 한국어 주석: 삼중 앙상블 모델 생성 및 캘리브레이션 (LightGBM + CatBoost + XGBoost)
def build_and_train_ensemble(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    # 한국어 주석: LightGBM, CatBoost, XGBoost 각각 학습 및 캘리브레이션
    members = []
    
    # validation set을 위한 분할 (early stopping용)
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.12, random_state=RANDOM_SEED, stratify=y_train
    )
    
    # 1. LightGBM
    lgb_model = create_lightgbm_model(seed=RANDOM_SEED)
    # Early stopping을 위한 validation set 사용
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
    )
    # Calibration을 위해 모델 복사 (early stopping 콜백 제거)
    from sklearn.base import clone
    lgb_for_calib = clone(lgb_model)
    lgb_for_calib.set_params(callbacks=None)  # 콜백 제거
    lgb_calib = create_calibrated_model(lgb_for_calib)
    lgb_calib.fit(X_train, y_train)
    members.append(lgb_calib)
    
    # 2. CatBoost
    # CatBoost는 class_weights를 딕셔너리로 계산
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    cat_class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cat_weight_dict = dict(zip(classes, cat_class_weights))
    cat_model = create_catboost_model(seed=RANDOM_SEED, class_weights=cat_weight_dict)
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    cat_calib = create_calibrated_model(cat_model)
    cat_calib.fit(X_train, y_train)
    members.append(cat_calib)
    
    # 3. XGBoost
    xgb_model = create_xgboost_model(seed=RANDOM_SEED)
    # XGBoost는 불균형 데이터를 위해 scale_pos_weight 계산
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes, class_weights))
    xgb_model.set_params(scale_pos_weight=weight_dict[1] / weight_dict[0])
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_calib = create_calibrated_model(xgb_model)
    xgb_calib.fit(X_train, y_train)
    members.append(xgb_calib)
    
    # 한국어 주석: 삼중 앙상블로 감싸서 반환
    ensemble = AvgProbaEnsemble(members)
    return ensemble


# 한국어 주석: A/B 모델 및 전처리기 저장
def save_model_artifacts(
    preproc_A: Any,
    ensemble_A: Any,
    preproc_B: Any,
    ensemble_B: Any,
) -> None:
    # 한국어 주석: 디렉터리가 없으면 생성
    os.makedirs(os.path.dirname(A_MODEL_FILE), exist_ok=True)
    
    # 한국어 주석: A/B 각각 저장
    joblib.dump(ensemble_A, A_MODEL_FILE)
    joblib.dump(preproc_A, A_PREPROC_FILE)
    joblib.dump(ensemble_B, B_MODEL_FILE)
    joblib.dump(preproc_B, B_PREPROC_FILE)
    
    # 한국어 주석: 메타데이터 저장
    meta = {
        "model": "LightGBM + CatBoost + XGBoost Ensemble + IsotonicCalibration",
        "lightgbm_params": BASE_LIGHTGBM_PARAMS,
        "catboost_params": BASE_CATBOOST_PARAMS,
        "xgboost_params": BASE_XGBOOST_PARAMS,
        "calib_method": CALIB_METHOD,
        "calib_cv": CALIBRATION_CV,
        "sklearn_version": sklver,
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# 한국어 주석: A/B 모델 로딩
def load_models() -> tuple:
    ensemble_A = joblib.load(A_MODEL_FILE)
    preproc_A = joblib.load(A_PREPROC_FILE)
    ensemble_B = joblib.load(B_MODEL_FILE)
    preproc_B = joblib.load(B_PREPROC_FILE)
    return preproc_A, ensemble_A, preproc_B, ensemble_B


# 한국어 주석: 통합 모델/전처리 로딩 (train.py 경로)
def load_combined_model() -> tuple:
    ensemble = joblib.load(MODEL_FILE)
    preproc = joblib.load(PREPROC_ALL_FILE)
    return preproc, ensemble


