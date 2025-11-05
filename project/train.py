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
    
    # 한국어 주석: 제거할 컬럼 결정
    drop_cols = [key, label_col] + (["Test"] if "Test" in df.columns else [])
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
        print(f"[{which}] Holdout AUC={auc:.5f}, Brier={brier:.5f}")
    except Exception as e:
        print(f"[{which}] validation logging skipped: {e}")
    
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
    
    # 한국어 주석: A 학습
    A_train_idx = train_idx[train_idx["Test"] == "A"].copy()
    print("[A] training path...")
    preproc_A, ensemble_A = fit_single_model(A_train_idx, A_train_feat, "Label", "A")
    
    # 한국어 주석: B 학습
    B_train_idx = train_idx[train_idx["Test"] == "B"].copy()
    print("[B] training path...")
    preproc_B, ensemble_B = fit_single_model(B_train_idx, B_train_feat, "Label", "B")
    
    # 한국어 주석: 모델 저장
    save_model_artifacts(preproc_A, ensemble_A, preproc_B, ensemble_B)
    print("[완료] 모델이 저장되었습니다.")


if __name__ == "__main__":
    main()

