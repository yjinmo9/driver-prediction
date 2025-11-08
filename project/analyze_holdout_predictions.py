#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Holdout 예측값 분포 분석 및 파라미터 조정 시뮬레이션"""

import os
import warnings
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
    preprocess_A_v2,
    preprocess_B_v2,
)
from src.model_utils import (
    build_and_train_ensemble,
    TemperatureScaler,
    CalibratedWithTemperature,
)
from src.evaluate import compute_ece, compute_final_score
import joblib

warnings.filterwarnings("ignore")


def analyze_predictions(y_true, y_pred, name="Model"):
    """예측값 분포 분석"""
    print(f"\n{'='*60}")
    print(f"[{name}] 예측값 분포 분석")
    print(f"{'='*60}")
    
    # 기본 통계
    print(f"\n📊 기본 통계:")
    print(f"  예측값 평균: {np.mean(y_pred):.6f}")
    print(f"  예측값 중앙값: {np.median(y_pred):.6f}")
    print(f"  예측값 표준편차: {np.std(y_pred):.6f}")
    print(f"  예측값 최소: {np.min(y_pred):.6f}")
    print(f"  예측값 최대: {np.max(y_pred):.6f}")
    print(f"  예측값 범위: {np.max(y_pred) - np.min(y_pred):.6f}")
    
    # 분위수
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n📈 분위수:")
    for p in percentiles:
        val = np.percentile(y_pred, p)
        print(f"  {p}%: {val:.6f}")
    
    # 실제 Label 분포
    print(f"\n🎯 실제 Label 분포:")
    print(f"  Label 평균: {np.mean(y_true):.6f}")
    print(f"  Label 비율 (0/1): {(y_true==0).sum()}/{((y_true==1).sum())} ({np.mean(y_true==0):.2%}/{np.mean(y_true==1):.2%})")
    
    # 예측값 vs 실제 Label
    print(f"\n📉 예측값 vs 실제 Label:")
    print(f"  예측값 평균 - Label 평균: {np.mean(y_pred) - np.mean(y_true):.6f}")
    
    # 평가 지표
    auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    ece = compute_ece(y_true, y_pred, n_bins=15)
    final = compute_final_score(auc, brier, ece)
    
    print(f"\n📊 평가 지표:")
    print(f"  AUC: {auc:.5f}")
    print(f"  Brier: {brier:.5f}")
    print(f"  ECE: {ece:.5f}")
    print(f"  Final: {final:.5f}")
    
    return {
        "mean": np.mean(y_pred),
        "median": np.median(y_pred),
        "std": np.std(y_pred),
        "min": np.min(y_pred),
        "max": np.max(y_pred),
        "range": np.max(y_pred) - np.min(y_pred),
        "auc": auc,
        "brier": brier,
        "ece": ece,
        "final": final,
    }


def simulate_parameter_adjustment(
    X_tr, y_tr, X_val, y_val, 
    base_params, adjustments, which="A"
):
    """파라미터 조정 시뮬레이션"""
    print(f"\n{'='*60}")
    print(f"[{which}] 파라미터 조정 시뮬레이션")
    print(f"{'='*60}")
    
    results = []
    
    for adj_name, adj_params in adjustments.items():
        print(f"\n🔧 조정: {adj_name}")
        print(f"   파라미터: {adj_params}")
        
        # 파라미터 병합
        params = base_params.copy()
        params.update(adj_params)
        
        # 모델 학습
        try:
            ensemble = build_and_train_ensemble(X_tr, y_tr, custom_params=params)
            
            # 예측
            val_proba = np.clip(ensemble.predict_proba(X_val)[:, 1], 1e-7, 1-1e-7)
            
            # 온도 스케일링
            temp = TemperatureScaler()
            temp.fit(y_val, val_proba)
            ensemble = CalibratedWithTemperature(ensemble, temp)
            val_proba = np.clip(ensemble.predict_proba(X_val)[:, 1], 1e-7, 1-1e-7)
            
            # 평가
            stats = analyze_predictions(y_val, val_proba, f"{which}_{adj_name}")
            stats["adj_name"] = adj_name
            stats["params"] = params.copy()
            results.append(stats)
            
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            continue
    
    return results


def main():
    print("="*60)
    print("Holdout 예측값 분포 분석 및 파라미터 조정 시뮬레이션")
    print("="*60)
    
    # 데이터 로드
    train_idx, _ = read_index_files()
    A_train_feat, B_train_feat = read_feature_files("train")
    
    # 피처 엔지니어링
    A_train_feat = preprocess_A_v2(A_train_feat)
    B_train_feat = preprocess_B_v2(B_train_feat)
    
    # A 모델 분석
    print("\n" + "="*60)
    print("A 모델 분석")
    print("="*60)
    
    A_idx = train_idx[train_idx["Test"] == "A"].copy()
    A_df = A_idx.merge(A_train_feat, on="Test_id", how="left", validate="1:1")
    
    if 'Test_x' in A_df.columns or 'Test_y' in A_df.columns:
        if 'Test_x' in A_df.columns:
            A_df.rename(columns={'Test_x': 'Test'}, inplace=True)
        A_df.drop(columns=[c for c in ['Test_y'] if c in A_df.columns], inplace=True, errors='ignore')
    A_df = A_df.loc[:, ~A_df.columns.duplicated()]
    
    drop_cols = ["Test_id", "Label"]
    drop_cols += [c for c in ['Test_x', 'Test_y'] if c in A_df.columns]
    drop_cols = [c for c in drop_cols if c in A_df.columns]
    
    feature_cols = [c for c in A_df.columns if c not in drop_cols]
    A_df = add_rowwise_features(A_df, feature_cols)
    
    X_A = A_df.drop(columns=drop_cols)
    y_A = A_df["Label"].astype(int).values
    
    X_A_tr, X_A_val, y_A_tr, y_A_val = train_test_split(
        X_A, y_A, test_size=VALID_SIZE, random_state=RANDOM_SEED, stratify=y_A
    )
    
    # 전처리
    preproc_A = build_preprocessor(A_df, feature_cols)
    X_A_tr_t = preproc_A.fit_transform(X_A_tr)
    X_A_val_t = preproc_A.transform(X_A_val)
    
    # 현재 모델 (더_공격적_3 파라미터 - 최근 적용한 것)
    params_current = {
        "learning_rate": 0.05,
        "max_iter": 1200,
        "max_depth": None,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 0.6,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 45,
        "class_weight": None,
    }
    
    print("\n[현재 모델] 더_공격적_3 파라미터")
    ensemble_A = build_and_train_ensemble(X_A_tr_t, y_A_tr, custom_params=params_current)
    val_proba_A = np.clip(ensemble_A.predict_proba(X_A_val_t)[:, 1], 1e-7, 1-1e-7)
    temp_A = TemperatureScaler()
    temp_A.fit(y_A_val, val_proba_A)
    ensemble_A = CalibratedWithTemperature(ensemble_A, temp_A)
    val_proba_A = np.clip(ensemble_A.predict_proba(X_A_val_t)[:, 1], 1e-7, 1-1e-7)
    
    stats_A = analyze_predictions(y_A_val, val_proba_A, "A_현재")
    
    # 파라미터 조정 시뮬레이션
    adjustments = {
        "더_공격적_1": {
            "learning_rate": 0.05,
            "class_weight": None,  # balanced 제거
        },
        "더_공격적_2": {
            "learning_rate": 0.06,
            "max_iter": 1200,
            "class_weight": None,
        },
        "더_공격적_3": {
            "learning_rate": 0.05,
            "max_iter": 1200,
            "min_samples_leaf": 30,  # 더 작게
            "class_weight": None,
        },
        "매우_공격적_1": {
            "learning_rate": 0.07,
            "max_iter": 1500,
            "min_samples_leaf": 20,
            "l2_regularization": 0.4,
            "class_weight": None,
        },
        "매우_공격적_2": {
            "learning_rate": 0.08,
            "max_iter": 1500,
            "min_samples_leaf": 15,
            "l2_regularization": 0.3,
            "max_leaf_nodes": 63,
            "class_weight": None,
        },
        "매우_공격적_3": {
            "learning_rate": 0.06,
            "max_iter": 1800,
            "min_samples_leaf": 10,
            "l2_regularization": 0.2,
            "max_leaf_nodes": 127,
            "class_weight": None,
        },
    }
    
    results_A = simulate_parameter_adjustment(
        X_A_tr_t, y_A_tr, X_A_val_t, y_A_val,
        params_current, adjustments, "A"
    )
    
    # B 모델 분석
    print("\n" + "="*60)
    print("B 모델 분석")
    print("="*60)
    
    B_idx = train_idx[train_idx["Test"] == "B"].copy()
    B_df = B_idx.merge(B_train_feat, on="Test_id", how="left", validate="1:1")
    
    if 'Test_x' in B_df.columns or 'Test_y' in B_df.columns:
        if 'Test_x' in B_df.columns:
            B_df.rename(columns={'Test_x': 'Test'}, inplace=True)
        B_df.drop(columns=[c for c in ['Test_y'] if c in B_df.columns], inplace=True, errors='ignore')
    B_df = B_df.loc[:, ~B_df.columns.duplicated()]
    
    drop_cols = ["Test_id", "Label"]
    drop_cols += [c for c in ['Test_x', 'Test_y'] if c in B_df.columns]
    drop_cols = [c for c in drop_cols if c in B_df.columns]
    
    feature_cols = [c for c in B_df.columns if c not in drop_cols]
    B_df = add_rowwise_features(B_df, feature_cols)
    
    X_B = B_df.drop(columns=drop_cols)
    y_B = B_df["Label"].astype(int).values
    
    X_B_tr, X_B_val, y_B_tr, y_B_val = train_test_split(
        X_B, y_B, test_size=VALID_SIZE, random_state=RANDOM_SEED, stratify=y_B
    )
    
    # 전처리
    preproc_B = build_preprocessor(B_df, feature_cols)
    X_B_tr_t = preproc_B.fit_transform(X_B_tr)
    X_B_val_t = preproc_B.transform(X_B_val)
    
    # 현재 모델 (더_공격적_3 파라미터 - 최근 적용한 것)
    params_B_current = {
        "learning_rate": 0.05,
        "max_iter": 1200,
        "max_depth": None,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 0.7,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 50,
        "class_weight": None,
    }
    
    print("\n[현재 모델] 더_공격적_3 파라미터")
    ensemble_B = build_and_train_ensemble(X_B_tr_t, y_B_tr, custom_params=params_B_current)
    val_proba_B = np.clip(ensemble_B.predict_proba(X_B_val_t)[:, 1], 1e-7, 1-1e-7)
    temp_B = TemperatureScaler()
    temp_B.fit(y_B_val, val_proba_B)
    ensemble_B = CalibratedWithTemperature(ensemble_B, temp_B)
    val_proba_B = np.clip(ensemble_B.predict_proba(X_B_val_t)[:, 1], 1e-7, 1-1e-7)
    
    stats_B = analyze_predictions(y_B_val, val_proba_B, "B_현재")
    
    # 파라미터 조정 시뮬레이션
    results_B = simulate_parameter_adjustment(
        X_B_tr_t, y_B_tr, X_B_val_t, y_B_val,
        params_B_current, adjustments, "B"
    )
    
    # 종합 리포트
    print("\n" + "="*60)
    print("종합 리포트")
    print("="*60)
    
    print("\n📊 A 모델 비교:")
    print(f"  현재 Final: {stats_A['final']:.5f}")
    for r in results_A:
        print(f"  {r['adj_name']}: {r['final']:.5f} (차이: {r['final'] - stats_A['final']:+.5f})")
    
    print("\n📊 B 모델 비교:")
    print(f"  현재 Final: {stats_B['final']:.5f}")
    for r in results_B:
        print(f"  {r['adj_name']}: {r['final']:.5f} (차이: {r['final'] - stats_B['final']:+.5f})")
    
    print("\n💡 권장사항:")
    print("  - 예측값 범위가 넓어지면 AUC가 개선될 수 있음")
    print("  - class_weight=None으로 변경하면 더 공격적으로 예측")
    print("  - learning_rate 증가로 더 빠르게 학습")
    print("  - min_samples_leaf 감소로 더 세밀한 분할")


if __name__ == "__main__":
    main()

