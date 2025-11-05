import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    MODEL_DIR,
    RANDOM_SEED,
    VALID_SIZE,
)
from src.data_utils import read_index_files, read_feature_files
from src.feature_engineer import (
    get_drop_columns,
    split_numeric_categorical,
    build_preprocessor,
    add_rowwise_features,
)
from src.model_utils import build_and_train_ensemble, save_model_artifacts
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
    ensemble = build_and_train_ensemble(X_tr_t, y_tr)
    
    # 한국어 주석: 검증 데이터로 평가
    try:
        val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
        auc = roc_auc_score(y_val, val_proba)
        brier = brier_score_loss(y_val, val_proba)
        ece = compute_ece(y_val, val_proba, n_bins=15)
        final_score = compute_final_score(auc, brier, ece)
        print(f"[{which}] Holdout AUC={auc:.5f}, Brier={brier:.5f}, ECE={ece:.5f}, Final={final_score:.5f}")
    except Exception as e:
        print(f"[{which}] validation logging skipped: {e}")
    
    return preproc, ensemble


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

    val_proba = np.clip(ensemble.predict_proba(X_val_t)[:, 1], 1e-7, 1-1e-7)
    auc = roc_auc_score(y_val, val_proba)
    brier = brier_score_loss(y_val, val_proba)
    ece = compute_ece(y_val, val_proba, n_bins=15)
    final_score = compute_final_score(auc, brier, ece)
    print(f"[ALL] Holdout AUC={auc:.5f}, Brier={brier:.5f}, ECE={ece:.5f}, Final={final_score:.5f}")

    return preproc, ensemble


def main() -> None:
    # 한국어 주석: 경고 소거
    warnings.filterwarnings("ignore")
    
    # 한국어 주석: 디렉터리 준비
    os.makedirs(MODEL_DIR, exist_ok=True)
    
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
    
    # 한국어 주석: 통합 학습 경로
    print("[ALL] training path...")
    preproc_all, ensemble_all = fit_combined_model(train_idx, A_train_feat, B_train_feat, "Label")

    # 한국어 주석: 단일 모델 저장 (통합)
    joblib.dump(ensemble_all, MODEL_FILE)
    joblib.dump(preproc_all, os.path.join(MODEL_DIR, "preproc_all.pkl"))
    print("[완료] 통합 모델이 저장되었습니다.")


if __name__ == "__main__":
    main()

