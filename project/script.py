import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DATA_DIR, OUTPUT_DIR, SUBMISSION_FILE, MODEL_DIR, MODEL_FILE
from src.data_utils import read_index_files, read_feature_files
from src.model_utils import load_models, load_combined_model
from src.feature_engineer import split_numeric_categorical, build_preprocessor, add_rowwise_features
from src.evaluate import rebalance_probabilities
import joblib


def safe_main() -> int:
    # 한국어 주석: 예외 발생 시 프로세스가 종료되지 않도록 안전 실행 래퍼
    try:
        return main()
    except Exception as e:
        # 한국어 주석: 대회 오프라인 채점 서버에서 크래시 방지를 위해 표준 오류로만 알림
        print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
        return 1


# 한국어 주석: A/B 중 하나에 대한 추론 함수
def predict_single_model(
    df_idx: pd.DataFrame,
    df_feat: pd.DataFrame,
    preproc,
    ensemble,
    which: str,
) -> pd.DataFrame:
    # 한국어 주석: 인덱스와 피처를 Test_id로 병합
    key = "Test_id"
    df = df_idx.merge(df_feat, on=key, how="left", validate="1:1")
    
    # 한국어 주석: 제거할 컬럼 결정
    drop_cols = [key] + (["Test"] if "Test" in df.columns else [])
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    # 한국어 주석: 피처 컬럼 추출
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # 한국어 주석: 행 단위 피처 추가 (학습 시와 동일하게)
    df = add_rowwise_features(df, feature_cols)
    
    # 한국어 주석: X 준비 및 전처리
    X = df.drop(columns=drop_cols, errors="ignore")
    X_t = preproc.transform(X)
    
    # 한국어 주석: 확률 예측 (0~1로 클리핑)
    proba = np.clip(ensemble.predict_proba(X_t)[:, 1], 1e-7, 1 - 1e-7)
    
    # 한국어 주석: 결과 반환
    out = df_idx[[key]].copy()
    out["Label"] = proba
    return out


# 한국어 주석: 통합 모델을 사용한 추론 함수 (리밸런싱 포함)
def predict_combined_model(
    df_idx: pd.DataFrame,
    A_feat: pd.DataFrame,
    B_feat: pd.DataFrame,
    preproc,
    ensemble,
    train_prior: float,
    test_prior: float = None,
    rebalance: bool = True,
) -> pd.DataFrame:
    """통합 모델로 추론하고 리밸런싱 적용"""
    key = "Test_id"
    
    # A/B 피처 통합
    all_feat = pd.concat([A_feat, B_feat], axis=0, ignore_index=True)
    
    # 인덱스와 병합
    df = df_idx.merge(all_feat, on=key, how="left", validate="1:1")
    
    # 제거할 컬럼 결정
    drop_cols = [key] + (["Test"] if "Test" in df.columns else [])
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    # 피처 컬럼 추출
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # 행 단위 피처 추가
    df = add_rowwise_features(df, feature_cols)
    
    # X 준비 및 전처리
    X = df.drop(columns=drop_cols, errors="ignore")
    X_t = preproc.transform(X)
    
    # 확률 예측
    proba = np.clip(ensemble.predict_proba(X_t)[:, 1], 1e-7, 1 - 1e-7)
    
    # 리밸런싱 적용 (옵션)
    if rebalance:
        proba = rebalance_probabilities(proba, train_prior, test_prior, method="bayes")
    
    # 결과 반환
    out = df_idx[[key]].copy()
    out["Label"] = proba
    return out


def main() -> int:
    warnings.filterwarnings("ignore")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 한국어 주석: 인덱스 파일 로드
    train_idx, test_idx = read_index_files()
    
    # 한국어 주석: A/B 상세 데이터 로드
    A_train_feat, B_train_feat = read_feature_files("train")
    A_test_feat, B_test_feat = read_feature_files("test")
    
    # 한국어 주석: 통합 모델 로드 (리밸런싱 포함)
    try:
        ensemble_all = joblib.load(MODEL_FILE)
        preproc_all = joblib.load(os.path.join(MODEL_DIR, "preproc_all.pkl"))
        train_prior = joblib.load(os.path.join(MODEL_DIR, "train_prior.pkl"))
        print(f"[INFO] 통합 모델 로드 완료 (Train prior: {train_prior:.4f})")
        
        # 통합 모델로 추론 (리밸런싱 적용)
        sub = predict_combined_model(
            test_idx, A_test_feat, B_test_feat,
            preproc_all, ensemble_all, train_prior,
            test_prior=None,  # 테스트 데이터의 실제 비율을 모르면 None (train_prior 사용)
            rebalance=True  # 리밸런싱 활성화
        )
    except FileNotFoundError:
        # 한국어 주석: 통합 모델이 없으면 A/B 분리 모델 사용 (기존 방식)
        print("[INFO] 통합 모델이 없어 A/B 분리 모델을 사용합니다.")
        preproc_A, ensemble_A, preproc_B, ensemble_B = load_models()
        
        # 한국어 주석: A 추론
        A_test_idx = test_idx[test_idx["Test"] == "A"].copy()
        preds_A = predict_single_model(A_test_idx, A_test_feat, preproc_A, ensemble_A, "A") if len(A_test_idx) else None
        
        # 한국어 주석: B 추론
        B_test_idx = test_idx[test_idx["Test"] == "B"].copy()
        preds_B = predict_single_model(B_test_idx, B_test_feat, preproc_B, ensemble_B, "B") if len(B_test_idx) else None
        
        # 한국어 주석: 결과 병합
        if preds_A is not None and preds_B is not None:
            sub = pd.concat([preds_A, preds_B], axis=0, ignore_index=True)
        elif preds_A is not None:
            sub = preds_A.copy()
        elif preds_B is not None:
            sub = preds_B.copy()
        else:
            # 한국어 주석: 예외 처리 - 추론 결과가 없으면 0.001로 채움
            sub = test_idx[["Test_id"]].copy()
            sub["Label"] = 0.001
    
    # 한국어 주석: sample_submission.csv가 있으면 컬럼 순서 맞추기
    try:
        sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
        sub = sub.merge(sample[["Test_id"]], on="Test_id", how="right")
        sub = sub[["Test_id", "Label"]]
    except Exception:
        sub = sub[["Test_id", "Label"]]
    
    # 한국어 주석: 제출 파일 저장
    sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"[완료] submission saved -> {SUBMISSION_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(safe_main())

