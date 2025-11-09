#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CatBoost + HGB 3중 앙상블 조합 최종 모델 학습 및 통합 점수 계산"""

import os
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

try:
    import catboost as cb
except ImportError:
    print("❌ CatBoost가 설치되어 있지 않습니다. pip install catboost")
    cb = None

from src.config import (
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
    VALID_SIZE,
    ENSEMBLE_SEEDS,
    A_MODEL_FILE,
    B_MODEL_FILE,
    A_PREPROC_FILE,
    B_PREPROC_FILE,
    META_FILE,
)
from src.data_utils import read_index_files, read_feature_files
from src.feature_engineer import (
    split_numeric_categorical,
    build_preprocessor,
    add_rowwise_features,
    preprocess_A_v2,
    preprocess_B_v2,
)
from src.evaluate import compute_ece, compute_final_score
from src.model_utils import TemperatureScaler, AvgProbaEnsemble, CalibratedWithTemperature
from sklearn import __version__ as sklver
import json

warnings.filterwarnings("ignore")


def create_calibrated_model(estimator):
    """CalibratedClassifierCV 생성 (sklearn 버전 호환)"""
    try:
        major, minor, *_ = map(int, sklver.split(".")[:2])
    except Exception:
        major, minor = 1, 4
    
    kw = dict(method="isotonic", cv=3)
    if (major, minor) >= (1, 4):
        return CalibratedClassifierCV(estimator=estimator, **kw)
    else:
        return CalibratedClassifierCV(base_estimator=estimator, **kw)


class CatBoostEnsemble:
    """CatBoost 3중 앙상블 모델 래퍼"""
    def __init__(self, models, temp_scaler=None):
        self.models = models
        self.temp_scaler = temp_scaler
    
    def predict_proba(self, X):
        """3개 모델의 예측 확률 평균"""
        probs = [model.predict_proba(X)[:, 1] for model in self.models]
        proba = np.mean(probs, axis=0)
        proba = np.clip(proba, 1e-7, 1-1e-7)
        
        # 온도 스케일링 적용
        if self.temp_scaler is not None:
            proba = self.temp_scaler.transform(proba)
        
        return np.stack([1 - proba, proba], axis=1)


class CombinedEnsemble:
    """CatBoost + HGB 조합 앙상블"""
    def __init__(self, catboost_ensemble, hgb_ensemble, catboost_weight=0.5):
        self.catboost_ensemble = catboost_ensemble
        self.hgb_ensemble = hgb_ensemble
        self.catboost_weight = catboost_weight
        self.hgb_weight = 1.0 - catboost_weight
    
    def predict_proba(self, X):
        """두 모델의 예측 확률 가중 평균"""
        catboost_proba = self.catboost_ensemble.predict_proba(X)[:, 1]
        hgb_proba = self.hgb_ensemble.predict_proba(X)[:, 1]
        
        combined_proba = (
            self.catboost_weight * catboost_proba +
            self.hgb_weight * hgb_proba
        )
        combined_proba = np.clip(combined_proba, 1e-7, 1-1e-7)
        
        return np.stack([1 - combined_proba, combined_proba], axis=1)


def train_catboost_model(
    X_tr, y_tr, X_val, y_val,
    catboost_params,
    which="",
    use_temperature=True
):
    """CatBoost 3중 앙상블 모델 학습 및 평가"""
    if cb is None:
        raise ImportError("CatBoost가 설치되어 있지 않습니다.")
    
    print(f"[{which}] CatBoost 3중 앙상블 학습 시작...")
    
    # 클래스 불균형 처리
    pos_count = np.sum(y_tr)
    neg_count = len(y_tr) - pos_count
    class_weight = None
    if neg_count > 0 and pos_count > 0:
        class_weight = [neg_count / pos_count, 1.0]
    
    # CatBoost Pool 생성
    train_pool = cb.Pool(X_tr, label=y_tr)
    val_pool = cb.Pool(X_val, label=y_val)
    
    # 3개 시드로 앙상블 학습
    models = []
    for seed in ENSEMBLE_SEEDS:
        params = catboost_params.copy()
        params['random_seed'] = seed
        params['verbose'] = False
        if class_weight is not None:
            params['class_weights'] = class_weight
        
        model = cb.CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            verbose=False,
        )
        models.append(model)
    
    # 검증 데이터 예측 (온도 스케일링 전)
    val_proba_raw = np.mean([model.predict_proba(X_val)[:, 1] for model in models], axis=0)
    val_proba_raw = np.clip(val_proba_raw, 1e-7, 1-1e-7)
    
    # 온도 스케일링
    temp_scaler = None
    if use_temperature:
        temp_scaler = TemperatureScaler()
        temp_scaler.fit(y_val, val_proba_raw)
        val_proba = temp_scaler.transform(val_proba_raw)
    else:
        val_proba = val_proba_raw
    
    # 앙상블 생성 (온도 스케일러 포함)
    ensemble = CatBoostEnsemble(models, temp_scaler)
    
    # 평가
    auc = roc_auc_score(y_val, val_proba)
    brier = brier_score_loss(y_val, val_proba)
    ece = compute_ece(y_val, val_proba, n_bins=15)
    final = compute_final_score(auc, brier, ece)
    
    print(f"[{which}] CatBoost Holdout AUC={auc:.5f}, Brier={brier:.5f}, ECE={ece:.5f}, Final={final:.5f}")
    
    return ensemble, y_val, val_proba


def train_hgb_model(
    X_tr, y_tr, X_val, y_val,
    hgb_params,
    which="",
    use_temperature=False,  # 최적 조합: Temperature 없음
    use_calibration=False   # 최적 조합: 캘리브레이션 없음
):
    """HGB 3중 앙상블 모델 학습 및 평가"""
    print(f"[{which}] HGB 3중 앙상블 학습 시작...")
    
    # 3개 시드로 앙상블 학습
    members = []
    for seed in ENSEMBLE_SEEDS:
        params = hgb_params.copy()
        params['random_state'] = seed
        
        # HGB 모델 생성
        base = HistGradientBoostingClassifier(**params)
        base.fit(X_tr, y_tr)
        
        # 캘리브레이션 적용 (최적 조합: 사용 안 함)
        if use_calibration:
            calib = create_calibrated_model(base)
            calib.fit(X_tr, y_tr)
            members.append(calib)
        else:
            members.append(base)
    
    # 앙상블 예측 (AvgProbaEnsemble 사용)
    ensemble = AvgProbaEnsemble(members)
    val_proba_raw = ensemble.predict_proba(X_val)[:, 1]
    val_proba_raw = np.clip(val_proba_raw, 1e-7, 1-1e-7)
    
    # 온도 스케일링 (최적 조합: 사용 안 함)
    temp_scaler = None
    if use_temperature:
        temp_scaler = TemperatureScaler()
        temp_scaler.fit(y_val, val_proba_raw)
        val_proba = temp_scaler.transform(val_proba_raw)
    else:
        val_proba = val_proba_raw
    
    # 평가
    auc = roc_auc_score(y_val, val_proba)
    brier = brier_score_loss(y_val, val_proba)
    ece = compute_ece(y_val, val_proba, n_bins=15)
    final = compute_final_score(auc, brier, ece)
    
    print(f"[{which}] HGB Holdout AUC={auc:.5f}, Brier={brier:.5f}, ECE={ece:.5f}, Final={final:.5f}")
    
    # 앙상블 생성 (온도 스케일러 포함)
    calibrated_ensemble = CalibratedWithTemperature(ensemble, temp_scaler)
    
    return calibrated_ensemble, y_val, val_proba


def prepare_data(train_idx, train_feat, test_type):
    """데이터 준비 및 전처리"""
    idx = train_idx[train_idx["Test"] == test_type].copy()
    df = idx.merge(train_feat, on="Test_id", how="left", validate="1:1")
    
    # 중복 컬럼명 제거
    if 'Test_x' in df.columns or 'Test_y' in df.columns:
        if 'Test_x' in df.columns:
            df.rename(columns={'Test_x': 'Test'}, inplace=True)
        df.drop(columns=[c for c in ['Test_y'] if c in df.columns], inplace=True, errors='ignore')
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 제거할 컬럼 결정
    drop_cols = ["Test_id", "Label"]
    drop_cols += [c for c in ['Test_x', 'Test_y'] if c in df.columns]
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    # 피처 컬럼 추출
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # 행 단위 피처 추가
    df = add_rowwise_features(df, feature_cols)
    
    # 전처리 파이프라인 생성
    preproc = build_preprocessor(df, feature_cols)
    
    # X, y 준비
    X = df.drop(columns=drop_cols)
    y = df["Label"].astype(int).values
    
    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VALID_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # 전처리 적용
    X_tr_t = preproc.fit_transform(X_tr)
    X_val_t = preproc.transform(X_val)
    
    return preproc, X_tr_t, X_val_t, y_tr, y_val


def save_model_artifacts(
    preproc_A, combined_ensemble_A,
    preproc_B, combined_ensemble_B,
    feature_cols_A=None,
    feature_cols_B=None,
):
    """A/B 모델 및 전처리기를 bundle 형태로 저장 (pickle.dump 사용)"""
    import pickle
    
    os.makedirs(os.path.dirname(A_MODEL_FILE), exist_ok=True)
    
    # A bundle 생성
    # CombinedEnsemble에서 CatBoost와 HGB 앙상블 추출
    catboost_ensemble_A = combined_ensemble_A.catboost_ensemble
    hgb_ensemble_A = combined_ensemble_A.hgb_ensemble
    
    # CatBoost temperature 추출
    cb_temp_A = 1.0
    if hasattr(catboost_ensemble_A, 'temp_scaler') and catboost_ensemble_A.temp_scaler is not None:
        cb_temp_A = float(catboost_ensemble_A.temp_scaler.T)
    
    # HGB temperature 추출
    hgb_temp_A = 1.0
    if hasattr(hgb_ensemble_A, 'temp_scaler') and hgb_ensemble_A.temp_scaler is not None:
        hgb_temp_A = float(hgb_ensemble_A.temp_scaler.T)
    elif hasattr(hgb_ensemble_A, 'base_ensemble'):
        # CalibratedWithTemperature인 경우
        if hasattr(hgb_ensemble_A, 'temp_scaler') and hgb_ensemble_A.temp_scaler is not None:
            hgb_temp_A = float(hgb_ensemble_A.temp_scaler.T)
    
    # CatBoost 모델 리스트 추출
    cb_models_A = catboost_ensemble_A.models
    
    # HGB 모델 리스트 추출
    if hasattr(hgb_ensemble_A, 'base_ensemble'):
        # CalibratedWithTemperature인 경우
        hgb_models_A = hgb_ensemble_A.base_ensemble.models
    elif hasattr(hgb_ensemble_A, 'models'):
        # AvgProbaEnsemble인 경우
        hgb_models_A = hgb_ensemble_A.models
    else:
        hgb_models_A = []
    
    # feature_cols 추출 (전처리기에 저장된 컬럼 정보 사용)
    if feature_cols_A is None:
        try:
            # ColumnTransformer에서 컬럼 정보 추출
            if hasattr(preproc_A, 'feature_names_in_'):
                feature_cols_A = list(preproc_A.feature_names_in_)
            else:
                feature_cols_A = None
        except:
            feature_cols_A = None
    
    bundle_A = {
        "preproc": preproc_A,
        "catboost_models": cb_models_A,
        "hgb_models": hgb_models_A,
        "catboost_temperature": cb_temp_A,
        "hgb_temperature": hgb_temp_A,
        "catboost_weight": combined_ensemble_A.catboost_weight,
        "hgb_weight": combined_ensemble_A.hgb_weight,
        "feature_cols": feature_cols_A,  # 나중에 사용할 수 있도록
    }
    
    with open(os.path.join(MODEL_DIR, "bundle_A.pkl"), "wb") as f:
        pickle.dump(bundle_A, f)
    
    # B bundle 생성
    catboost_ensemble_B = combined_ensemble_B.catboost_ensemble
    hgb_ensemble_B = combined_ensemble_B.hgb_ensemble
    
    # CatBoost temperature 추출
    cb_temp_B = 1.0
    if hasattr(catboost_ensemble_B, 'temp_scaler') and catboost_ensemble_B.temp_scaler is not None:
        cb_temp_B = float(catboost_ensemble_B.temp_scaler.T)
    
    # HGB temperature 추출
    # HGB는 CalibratedWithTemperature로 감싸져 있음
    hgb_temp_B = 1.0
    if hasattr(hgb_ensemble_B, 'temp_scaler'):
        if hgb_ensemble_B.temp_scaler is not None:
            hgb_temp_B = float(hgb_ensemble_B.temp_scaler.T)
    
    # CatBoost 모델 리스트 추출
    cb_models_B = catboost_ensemble_B.models
    
    # HGB 모델 리스트 추출
    if hasattr(hgb_ensemble_B, 'base_ensemble'):
        hgb_models_B = hgb_ensemble_B.base_ensemble.models
    elif hasattr(hgb_ensemble_B, 'models'):
        hgb_models_B = hgb_ensemble_B.models
    else:
        hgb_models_B = []
    
    # feature_cols 추출 (전처리기에 저장된 컬럼 정보 사용)
    if feature_cols_B is None:
        try:
            # ColumnTransformer에서 컬럼 정보 추출
            if hasattr(preproc_B, 'feature_names_in_'):
                feature_cols_B = list(preproc_B.feature_names_in_)
        except:
            feature_cols_B = None
    
    # B bundle 저장
    bundle_B = {
        "preproc": preproc_B,
        "catboost_models": cb_models_B,
        "hgb_models": hgb_models_B,
        "catboost_temperature": cb_temp_B,
        "hgb_temperature": hgb_temp_B,
        "catboost_weight": combined_ensemble_B.catboost_weight,
        "hgb_weight": combined_ensemble_B.hgb_weight,
        "feature_cols": feature_cols_B,  # 피처 컬럼 목록
    }
    
    with open(os.path.join(MODEL_DIR, "bundle_B.pkl"), "wb") as f:
        pickle.dump(bundle_B, f)
    
    # 메타데이터 저장
    meta = {
        "model": "CatBoost + HGB 조합 앙상블",
        "catboost_params": {
            "learning_rate": 0.06,
            "iterations": 1000,
            "depth": 8,
            "l2_leaf_reg": 0.5,
            "min_data_in_leaf": 20,
            "name": "더 공격적 (A/B 동일)",
        },
        "catboost_use_temperature": True,
        "A_hgb_params": {
            "learning_rate": 0.05,
            "max_iter": 1200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 0.6,
            "class_weight": None,
            "name": "A-HGB 더_공격적_3",
        },
        "B_hgb_params": {
            "learning_rate": 0.05,
            "max_iter": 1200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 0.7,
            "class_weight": None,
            "name": "B-HGB 더_공격적_3",
        },
        "hgb_use_calibration": False,
        "hgb_use_temperature": False,
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
        "catboost_weight": 0.3,
        "hgb_weight": 0.7,
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    print("="*60)
    print("CatBoost + HGB 조합 앙상블 최종 모델 학습")
    print("최적 설정:")
    print("  - CatBoost: 더 공격적 (A/B 동일)")
    print("  - HGB: 더_공격적_3 (캘리브레이션 없음, Temperature 없음)")
    print("  - CatBoost: Temperature 있음")
    print("  - 가중치: CatBoost 0.3, HGB 0.7")
    print("="*60)
    
    # 디렉터리 준비
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 데이터 로드
    train_idx, _ = read_index_files()
    A_train_feat, B_train_feat = read_feature_files("train")
    
    # 피처 엔지니어링
    A_train_feat = preprocess_A_v2(A_train_feat)
    B_train_feat = preprocess_B_v2(B_train_feat)
    
    # ========== CatBoost 파라미터 ==========
    # 최적 조합: 더 공격적 (현재 B) - A/B 모두 동일 파라미터 사용
    catboost_params = {
        "learning_rate": 0.06,
        "iterations": 1000,
        "depth": 8,
        "l2_leaf_reg": 0.5,
        "min_data_in_leaf": 20,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
        "random_strength": 1.0,
        "border_count": 254,
        "early_stopping_rounds": 50,
    }
    
    # A/B 모두 동일한 CatBoost 파라미터 사용
    catboost_params_A = catboost_params
    catboost_params_B = catboost_params
    
    # ========== HGB 파라미터 ==========
    # A 모델: 더_공격적_3 파라미터 (Final: 0.16381, AUC: 0.68410)
    hgb_params_A = {
        "learning_rate": 0.05,
        "max_iter": 1200,
        "max_depth": None,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 0.6,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 45,
        "class_weight": None,  # balanced 제거로 더 공격적 예측
    }
    
    # B 모델: 더_공격적_3 파라미터 (Final: 0.21703, AUC: 0.58654)
    hgb_params_B = {
        "learning_rate": 0.05,
        "max_iter": 1200,
        "max_depth": None,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 0.7,  # B만 0.7
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 50,  # B만 50
        "class_weight": None,  # balanced 제거로 더 공격적 예측
    }
    
    # 데이터 준비
    print("\n[데이터 준비] A/B 데이터 전처리 중...")
    preproc_A, X_A_tr, X_A_val, y_A_tr, y_A_val = prepare_data(
        train_idx, A_train_feat, "A"
    )
    preproc_B, X_B_tr, X_B_val, y_B_tr, y_B_val = prepare_data(
        train_idx, B_train_feat, "B"
    )
    
    # 가중치 고정: CatBoost 0.3, HGB 0.7 (최적 조합)
    CATBOOST_WEIGHT = 0.3
    HGB_WEIGHT = 0.7
    
    # A 모델 학습 (CatBoost + HGB 병렬)
    print("\n[학습 시작] A 모델 학습 중...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_cb_A = executor.submit(
            train_catboost_model,
            X_A_tr, y_A_tr, X_A_val, y_A_val,
            catboost_params_A,
            "A-CatBoost",
            use_temperature=True,  # CatBoost: Temperature 있음
        )
        future_hgb_A = executor.submit(
            train_hgb_model,
            X_A_tr, y_A_tr, X_A_val, y_A_val,
            hgb_params_A,
            "A-HGB",
            use_temperature=False,  # HGB: Temperature 없음
            use_calibration=False,  # HGB: 캘리브레이션 없음
        )
        
        catboost_ensemble_A, _, catboost_proba_A = future_cb_A.result()
        hgb_ensemble_A, _, hgb_proba_A = future_hgb_A.result()
    
    # B 모델 학습 (CatBoost + HGB 병렬)
    print("\n[학습 시작] B 모델 학습 중...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_cb_B = executor.submit(
            train_catboost_model,
            X_B_tr, y_B_tr, X_B_val, y_B_val,
            catboost_params_B,
            "B-CatBoost",
            use_temperature=True,  # CatBoost: Temperature 있음
        )
        future_hgb_B = executor.submit(
            train_hgb_model,
            X_B_tr, y_B_tr, X_B_val, y_B_val,
            hgb_params_B,
            "B-HGB",
            use_temperature=False,  # HGB: Temperature 없음
            use_calibration=False,  # HGB: 캘리브레이션 없음
        )
        
        catboost_ensemble_B, _, catboost_proba_B = future_cb_B.result()
        hgb_ensemble_B, _, hgb_proba_B = future_hgb_B.result()
    
    # 조합 앙상블 생성 및 평가
    print("\n[A] 조합 앙상블 테스트... (가중치: CatBoost 0.3, HGB 0.7)")
    combined_ensemble_A = CombinedEnsemble(catboost_ensemble_A, hgb_ensemble_A, catboost_weight=CATBOOST_WEIGHT)
    combined_proba_A = combined_ensemble_A.predict_proba(X_A_val)[:, 1]
    
    auc_A = roc_auc_score(y_A_val, combined_proba_A)
    brier_A = brier_score_loss(y_A_val, combined_proba_A)
    ece_A = compute_ece(y_A_val, combined_proba_A, n_bins=15)
    final_A = compute_final_score(auc_A, brier_A, ece_A)
    print(f"[A] 조합 Holdout AUC={auc_A:.5f}, Brier={brier_A:.5f}, ECE={ece_A:.5f}, Final={final_A:.5f}")
    
    print("\n[B] 조합 앙상블 테스트... (가중치: CatBoost 0.3, HGB 0.7)")
    combined_ensemble_B = CombinedEnsemble(catboost_ensemble_B, hgb_ensemble_B, catboost_weight=CATBOOST_WEIGHT)
    combined_proba_B = combined_ensemble_B.predict_proba(X_B_val)[:, 1]
    
    auc_B = roc_auc_score(y_B_val, combined_proba_B)
    brier_B = brier_score_loss(y_B_val, combined_proba_B)
    ece_B = compute_ece(y_B_val, combined_proba_B, n_bins=15)
    final_B = compute_final_score(auc_B, brier_B, ece_B)
    print(f"[B] 조합 Holdout AUC={auc_B:.5f}, Brier={brier_B:.5f}, ECE={ece_B:.5f}, Final={final_B:.5f}")
    
    # 통합 점수 계산 (A+B 합쳐서)
    print("\n" + "="*60)
    print("통합 점수 계산")
    print("="*60)
    y_val_all = np.concatenate([y_A_val, y_B_val])
    val_proba_all = np.concatenate([combined_proba_A, combined_proba_B])
    
    auc_all = roc_auc_score(y_val_all, val_proba_all)
    brier_all = brier_score_loss(y_val_all, val_proba_all)
    ece_all = compute_ece(y_val_all, val_proba_all, n_bins=15)
    final_all = compute_final_score(auc_all, brier_all, ece_all)
    
    print(f"[ALL] Holdout AUC={auc_all:.5f}, Brier={brier_all:.5f}, ECE={ece_all:.5f}, Final={final_all:.5f}")
    
    # A/B 개별 점수 요약
    print(f"\n[A] Holdout AUC={auc_A:.5f}, Brier={brier_A:.5f}, ECE={ece_A:.5f}, Final={final_A:.5f}")
    print(f"[B] Holdout AUC={auc_B:.5f}, Brier={brier_B:.5f}, ECE={ece_B:.5f}, Final={final_B:.5f}")
    
    # 모델 저장
    print("\n[저장] 모델 및 전처리기 저장 중...")
    save_model_artifacts(
        preproc_A, combined_ensemble_A,
        preproc_B, combined_ensemble_B,
    )
    print("[완료] A/B CatBoost + HGB 조합 앙상블 모델이 저장되었습니다.")
    print(f"\n최종 통합 점수: {final_all:.5f}")


if __name__ == "__main__":
    main()

