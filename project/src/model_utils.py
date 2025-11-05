import json
import os
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import __version__ as sklver

from .config import (
    A_MODEL_FILE,
    B_MODEL_FILE,
    A_PREPROC_FILE,
    B_PREPROC_FILE,
    META_FILE,
    ENSEMBLE_SEEDS,
    BASE_HGB_PARAMS,
    CALIB_METHOD,
    CALIBRATION_CV,
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


# 한국어 주석: 여러 시드로 앙상블 모델 생성 및 캘리브레이션
def build_and_train_ensemble(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    # 한국어 주석: 3개의 다른 시드로 각각 학습 및 캘리브레이션
    members = []
    for sd in ENSEMBLE_SEEDS:
        base = create_base_estimator(seed=sd)
        base.fit(X_train, y_train)
        calib = create_calibrated_model(base)
        calib.fit(X_train, y_train)
        members.append(calib)
    
    # 한국어 주석: 앙상블로 감싸서 반환
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
        "model": "HistGradientBoostingClassifier + IsotonicCalibration",
        "hgb_params": BASE_HGB_PARAMS,
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
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


