import os
import warnings
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    MODEL_DIR,
    RANDOM_SEED,
    VALID_SIZE,
    OUTPUT_DIR,
)
from src.data_utils import read_index_files, read_feature_files
from src.feature_engineer import (
    get_drop_columns,
    split_numeric_categorical,
    build_preprocessor,
    add_rowwise_features,
)
from src.model_utils import (
    build_and_train_ensemble,
    save_model_artifacts,
    TemperatureScaler,
    CalibratedWithTemperature,
)
from src.feature_engineer import preprocess_A_v2, preprocess_B_v2
from src.evaluate import compute_ece, compute_final_score
from src.config import MODEL_FILE
import joblib


# 한국어 주석: A/B 중 하나에 대한 모델 학습 함수
def fit_single_model(
    df_idx: pd.DataFrame,
    df_feat: pd.DataFrame,
    label_col: str,
    which: str,
    custom_params: dict = None,
) -> Tuple:
    # 한국어 주석: 인덱스와 피처를 Test_id로 병합
    key = "Test_id"
    assert key in df_feat.columns, f"{which}: '{key}' not found in features"
    
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    # 중복 컬럼명 제거 및 Test 중복 해소
    if 'Test_x' in df.columns or 'Test_y' in df.columns:
        # 우선순위: df_idx의 Test를 유지
        if 'Test_x' in df.columns:
            df.rename(columns={'Test_x': 'Test'}, inplace=True)
        df.drop(columns=[c for c in ['Test_y'] if c in df.columns], inplace=True, errors='ignore')
    # 완전 동일한 이름의 중복 컬럼 제거
    df = df.loc[:, ~df.columns.duplicated()]

    # 한국어 주석: 제거할 컬럼 결정
    drop_cols = [key, label_col]
    # Test는 피처로 사용 (분포 차이를 학습하도록 남김)
    drop_cols += [c for c in ['Test_x', 'Test_y'] if c in df.columns]
    drop_cols = [c for c in drop_cols if c in df.columns]  # 한국어 주석: 존재하는 것만
    
    # 한국어 주석: 피처 컬럼 추출
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # 한국어 주석: 행 단위 피처 추가 (결측치 개수/비율)
    df = add_rowwise_features(df, feature_cols)
    
    # 한국어 주석: 수치/범주형 분리
    num_cols, cat_cols = split_numeric_categorical(df, feature_cols)
    
    # 한국어 주석: 전처리 파이프라인 생성
    preproc = build_preprocessor(df, feature_cols)
    
    # 한국어 주석: X, y 준비
    X = df.drop(columns=drop_cols)
    y = df[label_col].astype(int).values
    
    # 한국어 주석: 학습/검증 분할
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VALID_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # 한국어 주석: 전처리 적용
    X_tr_t = preproc.fit_transform(X_tr)
    X_val_t = preproc.transform(X_val)
    
    # 한국어 주석: 앙상블 학습 (3개 시드로 학습 후 평균)
    ensemble = build_and_train_ensemble(X_tr_t, y_tr, custom_params=custom_params)
    
    # 한국어 주석: 검증 데이터로 평가 및 온도 스케일링
    try:
        # 기본 캘리브레이션(Platt/Isotonic) 후 확률
        val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
        # 온도 스케일링으로 ECE/Brier 추가 보정 (밸리데이션 기반)
        temp = TemperatureScaler()
        temp.fit(y_val, val_proba)
        ensemble = CalibratedWithTemperature(ensemble, temp)
        val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
    except Exception as e:
        print(f"[{which}] validation processing skipped: {e}")
        val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
    
    return preproc, ensemble, y_val, val_proba


# 한국어 주석: A+B 통합 데이터로 하나의 모델 학습
def fit_combined_model(
    train_idx: pd.DataFrame,
    A_feat: pd.DataFrame,
    B_feat: pd.DataFrame,
    label_col: str,
):
    key = 'Test_id'
    assert key in A_feat.columns and key in B_feat.columns, 'Test_id missing in features'

    # 통합 피처셋
    all_feat = pd.concat([A_feat, B_feat], axis=0, ignore_index=True)

    # 인덱스와 병합
    df = train_idx.merge(all_feat, on=key, how='left', validate='1:1')

    # Test_x/Test_y 정리 및 중복 컬럼 제거
    if 'Test_x' in df.columns or 'Test_y' in df.columns:
        if 'Test_x' in df.columns:
            df.rename(columns={'Test_x': 'Test'}, inplace=True)
        df.drop(columns=[c for c in ['Test_y'] if c in df.columns], inplace=True, errors='ignore')
    df = df.loc[:, ~df.columns.duplicated()]

    # 제거 컬럼
    drop_cols = [key, label_col]
    # Test는 피처로 사용
    drop_cols += [c for c in ['Test_x', 'Test_y'] if c in df.columns]
    drop_cols = [c for c in drop_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    df = add_rowwise_features(df, feature_cols)
    preproc = build_preprocessor(df, feature_cols)

    X = df.drop(columns=drop_cols)
    y = df[label_col].astype(int).values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VALID_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    X_tr_t = preproc.fit_transform(X_tr)
    X_val_t = preproc.transform(X_val)

    ensemble = build_and_train_ensemble(X_tr_t, y_tr)

    # 온도 스케일링 보정
    val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
    temp = TemperatureScaler()
    temp.fit(y_val, val_proba)
    ensemble = CalibratedWithTemperature(ensemble, temp)
    val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
    auc = roc_auc_score(y_val, val_proba)
    brier = brier_score_loss(y_val, val_proba)
    ece = compute_ece(y_val, val_proba, n_bins=15)
    final_score = compute_final_score(auc, brier, ece)
    print(f"[ALL] Holdout AUC={auc:.5f}, Brier={brier:.5f}, ECE={ece:.5f}, Final={final_score:.5f}")

    # 한국어 주석: 학습 데이터의 클래스 비율 저장 (리밸런싱용)
    train_prior = float(np.mean(y_tr))
    
    return preproc, ensemble, train_prior


def main() -> None:
    # 한국어 주석: 경고 소거
    warnings.filterwarnings("ignore")
    
    # 한국어 주석: 디렉터리 준비
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 한국어 주석: 인덱스 파일 로드
    train_idx, test_idx = read_index_files()
    
    # 한국어 주석: A/B 상세 데이터 로드
    A_train_feat, B_train_feat = read_feature_files("train")
    A_test_feat, B_test_feat = read_feature_files("test")

    # v2 피처 엔지니어링 적용 (시퀀스 요약 등)
    try:
        A_train_feat = preprocess_A_v2(A_train_feat)
        B_train_feat = preprocess_B_v2(B_train_feat)
        A_test_feat = preprocess_A_v2(A_test_feat)
        B_test_feat = preprocess_B_v2(B_test_feat)
        print("[v2] feature engineering applied to A/B train & test features")
    except Exception as e:
        print(f"[v2] feature engineering skipped due to error: {e}")
    
    # 한국어 주석: A/B 각각 다른 파라미터로 병렬 학습
    # 분석 결과 기반 최적 조정 (더 공격적 예측)
    # A 모델: 더_공격적_3 (Final 0.16381, AUC 0.68410)
    params_A_optimized = {
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
    # B 모델: 더_공격적_3 (Final 0.21703, AUC 0.58654)
    params_B_optimized = {
        "learning_rate": 0.05,
        "max_iter": 1200,
        "max_depth": None,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 0.7,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 50,
        "class_weight": None,  # balanced 제거로 더 공격적 예측
    }
    
    print("[A] training path (최적 조정 파라미터: 더 공격적 예측)...")
    print("[B] training path (최적 조정 파라미터: 더 공격적 예측)...")
    
    # 병렬 처리로 A/B 동시 학습
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_A = executor.submit(
            fit_single_model,
            train_idx[train_idx["Test"] == "A"].copy(),
            A_train_feat,
            "Label",
            "A",
            params_A_optimized,
        )
        future_B = executor.submit(
            fit_single_model,
            train_idx[train_idx["Test"] == "B"].copy(),
            B_train_feat,
            "Label",
            "B",
            params_B_optimized,
        )
        
        preproc_A, ensemble_A, y_val_A, val_proba_A = future_A.result()
        preproc_B, ensemble_B, y_val_B, val_proba_B = future_B.result()
    
    # 한국어 주석: 전체 종합 점수 계산 (A+B 합쳐서)
    y_val_all = np.concatenate([y_val_A, y_val_B])
    val_proba_all = np.concatenate([val_proba_A, val_proba_B])
    auc_all = roc_auc_score(y_val_all, val_proba_all)
    brier_all = brier_score_loss(y_val_all, val_proba_all)
    ece_all = compute_ece(y_val_all, val_proba_all, n_bins=15)
    final_all = compute_final_score(auc_all, brier_all, ece_all)
    print(f"[ALL] Holdout AUC={auc_all:.5f}, Brier={brier_all:.5f}, ECE={ece_all:.5f}, Final={final_all:.5f}")
    
    # 한국어 주석: A/B 모델 및 전처리 저장
    save_model_artifacts(preproc_A, ensemble_A, preproc_B, ensemble_B)
    print("[완료] A/B 분리 모델이 저장되었습니다. (최적 조정 파라미터: 더 공격적 예측)")


if __name__ == "__main__":
    main()

