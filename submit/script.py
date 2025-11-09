#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CatBoost + HGB 조합 앙상블 모델 제출 스크립트 (pickle.load 사용)"""

import os
import pickle
import numpy as np
import pandas as pd
import sys

# numpy.random 모듈을 먼저 import (pickle 호환성을 위해)
import numpy.random
try:
    import numpy.random._pickle
    import numpy.random._pcg64
except ImportError:
    pass

# =============================================================
# 피처 엔지니어링 함수 (기존 코드에서 가져옴)
# =============================================================

def _seq_mean(series: pd.Series) -> pd.Series:
    return series.fillna("").apply(
        lambda x: np.fromstring(str(x), sep=",").mean() if str(x) else np.nan
    )


def _seq_std(series: pd.Series) -> pd.Series:
    return series.fillna("").apply(
        lambda x: np.fromstring(str(x), sep=",").std() if str(x) else np.nan
    )


def _seq_rate(series: pd.Series, target: str = "1") -> pd.Series:
    target = str(target)

    def _count_rate(x: str) -> float:
        x = str(x)
        if not x:
            return np.nan
        parts = x.split(",")
        denom = len(parts)
        if denom == 0:
            return np.nan
        num = sum(1 for p in parts if p.strip() == target)
        return num / denom

    return series.fillna("").apply(_count_rate)


def _masked_mean_from_csv_series(cond_series: pd.Series, val_series: pd.Series, mask_val: float) -> pd.Series:
    def _helper(row):
        cond_str = str(row[0]) if pd.notna(row[0]) else ""
        val_str = str(row[1]) if pd.notna(row[1]) else ""
        if not cond_str or not val_str:
            return np.nan
        try:
            conds = np.fromstring(cond_str, sep=",")
            vals = np.fromstring(val_str, sep=",")
            if len(conds) != len(vals):
                return np.nan
            masked = vals[conds == mask_val]
            return masked.mean() if len(masked) > 0 else np.nan
        except:
            return np.nan

    df_tmp = pd.DataFrame({"cond": cond_series, "val": val_series})
    return df_tmp.apply(_helper, axis=1)


def _masked_mean_any_from_csv_series(cond_series: pd.Series, val_series: pd.Series, mask_vals: list) -> pd.Series:
    """여러 mask_val 중 하나라도 해당하는 값들의 평균"""
    def _helper(row):
        cond_str = str(row[0]) if pd.notna(row[0]) else ""
        val_str = str(row[1]) if pd.notna(row[1]) else ""
        if not cond_str or not val_str:
            return np.nan
        try:
            conds = np.fromstring(cond_str, sep=",")
            vals = np.fromstring(val_str, sep=",")
            if len(conds) != len(vals):
                return np.nan
            mask = np.isin(conds, mask_vals)
            masked = vals[mask]
            return masked.mean() if len(masked) > 0 else np.nan
        except:
            return np.nan

    df_tmp = pd.DataFrame({"cond": cond_series, "val": val_series})
    return df_tmp.apply(_helper, axis=1)


def preprocess_A_v2(train_A: pd.DataFrame) -> pd.DataFrame:
    """A 데이터 피처 엔지니어링(v2)"""
    df = train_A.copy()
    df = df.dropna().reset_index(drop=True)

    drop_cols = [
        "A1-1", "A1-2",
        "A3-1", "A3-2", "A3-3", "A3-4", "A3-5",
        "A4-1", "A4-2", "A4-3",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    feats = pd.DataFrame(index=df.index)

    if "A1-3" in df.columns:
        feats["A1_resp_rate"] = _seq_rate(df["A1-3"], "0")
    if "A1-4" in df.columns:
        feats["A1_rt_mean"] = _seq_mean(df["A1-4"])
        feats["A1_rt_std"] = _seq_std(df["A1-4"])

    if "A2-3" in df.columns:
        feats["A2_resp_rate"] = _seq_rate(df["A2-3"], "0")
    if "A2-4" in df.columns:
        feats["A2_rt_mean"] = _seq_mean(df["A2-4"])
        feats["A2_rt_std"] = _seq_std(df["A2-4"])

    if "A3-6" in df.columns:
        feats["A3_resp_rate"] = _seq_rate(df["A3-6"], "0")
    if "A3-7" in df.columns:
        feats["A3_rt_mean"] = _seq_mean(df["A3-7"])
        feats["A3_rt_std"] = _seq_std(df["A3-7"])

    if "A4-4" in df.columns:
        feats["A4_resp_rate"] = _seq_rate(df["A4-4"], "0")
    if "A4-3" in df.columns:
        feats["A4_acc_rate"] = _seq_rate(df["A4-3"], "1")
    if "A4-5" in df.columns:
        feats["A4_rt_mean"] = _seq_mean(df["A4-5"])
        feats["A4_rt_std"] = _seq_std(df["A4-5"])
    if set(["A4-1", "A4-5"]).issubset(df.columns):
        feats["A4_stroop_diff"] = _masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 2) - \
            _masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 1)

    if "A5-3" in df.columns:
        feats["A5_resp_rate"] = _seq_rate(df["A5-3"], "0")
    if "A5-2" in df.columns:
        feats["A5_acc_rate"] = _seq_rate(df["A5-2"], "1")

    for col in [
        "A6-1", "A7-1", "A8-1", "A8-2",
        "A9-1", "A9-2", "A9-3", "A9-4", "A9-5",
    ]:
        if col in df.columns:
            feats[col] = df[col]

    base = df.drop(columns=[c for c in df.columns if c in feats.columns], errors="ignore")
    out = pd.concat([base, feats], axis=1)
    return out


def preprocess_B_v2(train_B: pd.DataFrame) -> pd.DataFrame:
    """B 데이터 피처 엔지니어링(v2)"""
    df = train_B.copy()
    df = df.dropna().reset_index(drop=True)

    feats = pd.DataFrame(index=df.index)

    if "B1-1" in df.columns:
        feats["B1_acc_rate"] = _seq_rate(df["B1-1"], "1")
    if "B1-2" in df.columns:
        feats["B1_rt_mean"] = _seq_mean(df["B1-2"])
        feats["B1_rt_std"] = _seq_std(df["B1-2"])

    if "B2-1" in df.columns:
        feats["B2_acc_rate"] = _seq_rate(df["B2-1"], "1")
    if "B2-2" in df.columns:
        feats["B2_rt_mean"] = _seq_mean(df["B2-2"])
        feats["B2_rt_std"] = _seq_std(df["B2-2"])

    if set(["B1_rt_mean", "B2_rt_mean"]).issubset(feats.columns):
        feats["B12_rt_diff"] = feats["B2_rt_mean"] - feats["B1_rt_mean"]
    if set(["B1_acc_rate", "B2_acc_rate"]).issubset(feats.columns):
        feats["B12_acc_diff"] = feats["B2_acc_rate"] - feats["B1_acc_rate"]

    for k in ["B3", "B4", "B5"]:
        acc_col, rt_col = f"{k}-1", f"{k}-2"
        if acc_col in df.columns:
            feats[f"{k}_acc_rate"] = _seq_rate(df[acc_col], "1")
        if rt_col in df.columns:
            feats[f"{k}_rt_mean"] = _seq_mean(df[rt_col])
            feats[f"{k}_rt_std"] = _seq_std(df[rt_col])

    # B4 난이도: B4-1에서 1,2 < 3,4,5,6 (hard)
    if set(["B4-1", "B4-2"]).issubset(df.columns):
        hard_mean = _masked_mean_any_from_csv_series(df["B4-1"], df["B4-2"], [3,4,5,6])
        easy_mean = _masked_mean_any_from_csv_series(df["B4-1"], df["B4-2"], [1,2])
        feats["B4_rt_hard_mean"] = hard_mean
        feats["B4_rt_easy_mean"] = easy_mean
        feats["B4_rt_hard_minus_easy"] = hard_mean - easy_mean
        # 정답이 1 vs 3/5 가정: 비율 비교(참고용)
        feats["B4_rate_ans1"] = _seq_rate(df["B4-1"], "1")
        feats["B4_rate_ans3"] = _seq_rate(df["B4-1"], "3")
        feats["B4_rate_ans5"] = _seq_rate(df["B4-1"], "5")
        feats["B4_rate_ans35"] = feats["B4_rate_ans3"].fillna(0.0) + feats["B4_rate_ans5"].fillna(0.0)

    # ---- B6~B8 ----
    for k in ["B6", "B7", "B8"]:
        if k in df.columns:
            feats[f"{k}_acc_rate"] = _seq_rate(df[k], "1")

    # B6 vs B7 난이도 차이: B7이 더 어려움 가정
    if set(["B6_acc_rate", "B7_acc_rate"]).issubset(feats.columns):
        feats["B76_acc_diff"] = feats["B7_acc_rate"] - feats["B6_acc_rate"]

    # ---- B9/B10 그룹 ----
    # B9: 1~4 한 묶음, 5 별도
    b9_exist = [c for c in ["B9-1","B9-2","B9-3","B9-4"] if c in df.columns]
    if b9_exist:
        feats["B9_g1to4_mean"] = df[b9_exist].mean(axis=1)
        feats["B9_g1to4_std"] = df[b9_exist].std(axis=1)
    if "B9-5" in df.columns:
        feats["B9_g5"] = df["B9-5"]

    # B10: 1~4 한 묶음, 5~6 한 묶음 (B10이 B9보다 더 어려움)
    b10_1to4 = [c for c in ["B10-1","B10-2","B10-3","B10-4"] if c in df.columns]
    if b10_1to4:
        feats["B10_g1to4_mean"] = df[b10_1to4].mean(axis=1)
        feats["B10_g1to4_std"] = df[b10_1to4].std(axis=1)
    b10_5to6 = [c for c in ["B10-5","B10-6"] if c in df.columns]
    if b10_5to6:
        feats["B10_g5to6_mean"] = df[b10_5to6].mean(axis=1)
        feats["B10_g5to6_std"] = df[b10_5to6].std(axis=1)

    # B10 vs B9 델타(난이도 차이 특징)
    if "B10_g1to4_mean" in feats.columns and "B9_g1to4_mean" in feats.columns:
        feats["B10minusB9_g1to4_mean"] = feats["B10_g1to4_mean"] - feats["B9_g1to4_mean"]
    if "B10_g1to4_std" in feats.columns and "B9_g1to4_std" in feats.columns:
        feats["B10minusB9_g1to4_std"] = feats["B10_g1to4_std"] - feats["B9_g1to4_std"]

    base = df.drop(columns=[c for c in df.columns if c in feats.columns], errors="ignore")
    out = pd.concat([base, feats], axis=1)
    return out


def add_rowwise_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """행 단위 피처 추가"""
    X = df[feature_cols]
    na_count = X.isna().sum(axis=1).astype(np.int32)
    na_ratio = (na_count / (len(feature_cols) + 1e-9)).astype(np.float32)
    df2 = df.copy()
    df2["NA_COUNT"] = na_count
    df2["NA_RATIO"] = na_ratio
    return df2


def predict_with_bundle(bundle, feat_df: pd.DataFrame) -> np.ndarray:
    """
    bundle로부터 예측 수행
    bundle 구조:
    - preproc: 전처리기
    - catboost_models: CatBoost 모델 리스트
    - hgb_models: HGB 모델 리스트
    - catboost_temperature: CatBoost temperature
    - hgb_temperature: HGB temperature
    - catboost_weight: CatBoost 가중치
    - hgb_weight: HGB 가중치
    """
    preproc = bundle["preproc"]
    catboost_models = bundle["catboost_models"]
    hgb_models = bundle["hgb_models"]
    cb_temp = bundle.get("catboost_temperature", 1.0)
    hgb_temp = bundle.get("hgb_temperature", 1.0)
    cb_weight = bundle.get("catboost_weight", 0.3)
    hgb_weight = bundle.get("hgb_weight", 0.7)

    # 전처리 적용
    drop_cols = ["Test_id", "Label"]
    drop_cols = [c for c in drop_cols if c in feat_df.columns]
    feature_cols = [c for c in feat_df.columns if c not in drop_cols]
    
    # 행 단위 피처 추가
    feat_df = add_rowwise_features(feat_df, feature_cols)
    
    X = feat_df.drop(columns=drop_cols, errors="ignore")
    X_t = preproc.transform(X)
    
    n = len(X_t)

    # CatBoost 예측
    catboost_probs = []
    for model in catboost_models:
        prob = model.predict_proba(X_t)[:, 1]
        catboost_probs.append(prob)
    catboost_mean = np.mean(catboost_probs, axis=0)

    # Temperature scaling for CatBoost
    if cb_temp != 1.0:
        # Temperature scaling: p' = sigmoid(logit(p) / T)
        catboost_mean = np.clip(catboost_mean, 1e-7, 1 - 1e-7)
        logits = np.log(catboost_mean / (1 - catboost_mean))
        catboost_mean = 1.0 / (1.0 + np.exp(-logits / cb_temp))
        catboost_mean = np.clip(catboost_mean, 1e-7, 1 - 1e-7)

    # HGB 예측
    hgb_probs = []
    for model in hgb_models:
        prob = model.predict_proba(X_t)[:, 1]
        hgb_probs.append(prob)
    hgb_mean = np.mean(hgb_probs, axis=0)

    # Temperature scaling for HGB
    if hgb_temp != 1.0:
        hgb_mean = np.clip(hgb_mean, 1e-7, 1 - 1e-7)
        logits = np.log(hgb_mean / (1 - hgb_mean))
        hgb_mean = 1.0 / (1.0 + np.exp(-logits / hgb_temp))
        hgb_mean = np.clip(hgb_mean, 1e-7, 1 - 1e-7)

    # 가중 평균
    blended = cb_weight * catboost_mean + hgb_weight * hgb_mean
    blended = np.clip(blended, 1e-7, 1 - 1e-7)

    return blended


def load_bundle(path):
    """bundle 로드 (pickle.load 사용)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    os.makedirs("output", exist_ok=True)

    # ------------------------------------------------
    # 0. 모델 로드 (pickle.load 사용)
    # ------------------------------------------------
    model_dir = "model"
    bundle_a_path = os.path.join(model_dir, "bundle_A.pkl")
    bundle_b_path = os.path.join(model_dir, "bundle_B.pkl")

    if not os.path.exists(bundle_a_path) or not os.path.exists(bundle_b_path):
        raise FileNotFoundError("bundle_A.pkl 혹은 bundle_B.pkl 이 없습니다.")

    bundle_a = load_bundle(bundle_a_path)
    bundle_b = load_bundle(bundle_b_path)

    # ------------------------------------------------
    # 1. base test 로드 (A/B 구분용)
    # ------------------------------------------------
    base_test = pd.read_csv("data/test.csv")
    if "Test" not in base_test.columns:
        raise ValueError("data/test.csv 에 'Test' 컬럼이 없습니다. A/B 구분을 할 수 없습니다.")

    # raw A/B
    has_a_raw = os.path.exists("data/test/A.csv")
    has_b_raw = os.path.exists("data/test/B.csv")

    if has_a_raw:
        a_raw = pd.read_csv("data/test/A.csv")
    else:
        a_raw = None

    if has_b_raw:
        b_raw = pd.read_csv("data/test/B.csv")
    else:
        b_raw = None

    # ------------------------------------------------
    # 2. A 처리
    # ------------------------------------------------
    test_a_ids = base_test[base_test["Test"] == "A"]["Test_id"].tolist()

    if a_raw is not None and len(test_a_ids) > 0:
        a_raw = a_raw[a_raw["Test_id"].isin(test_a_ids)].reset_index(drop=True)
        feat_a = preprocess_A_v2(a_raw)
        # Test_id가 이미 있으면 유지, 없으면 추가
        if "Test_id" not in feat_a.columns:
            feat_a.insert(0, "Test_id", a_raw["Test_id"].values)
        a_probs = predict_with_bundle(bundle_a, feat_a)
        a_pred_df = pd.DataFrame({"Test_id": feat_a["Test_id"], "Label": a_probs})
    else:
        a_pred_df = pd.DataFrame(
            {"Test_id": test_a_ids, "Label": np.full(len(test_a_ids), 0.5)}
        )
        print("[script] data/test/A.csv 없음 → A는 0.5로 채웠습니다.")

    # ------------------------------------------------
    # 3. B 처리
    # ------------------------------------------------
    test_b_ids = base_test[base_test["Test"] == "B"]["Test_id"].tolist()

    if b_raw is not None and len(test_b_ids) > 0:
        b_raw = b_raw[b_raw["Test_id"].isin(test_b_ids)].reset_index(drop=True)
        feat_b = preprocess_B_v2(b_raw)
        # Test_id가 이미 있으면 유지, 없으면 추가
        if "Test_id" not in feat_b.columns:
            feat_b.insert(0, "Test_id", b_raw["Test_id"].values)
        b_probs = predict_with_bundle(bundle_b, feat_b)
        b_pred_df = pd.DataFrame({"Test_id": feat_b["Test_id"], "Label": b_probs})
    else:
        b_pred_df = pd.DataFrame(
            {"Test_id": test_b_ids, "Label": np.full(len(test_b_ids), 0.5)}
        )
        print("[script] data/test/B.csv 없음 → B는 0.5로 채웠습니다.")

    # ------------------------------------------------
    # 4. 합치기 + base_test 순서로 정렬
    # ------------------------------------------------
    sub = pd.concat([a_pred_df, b_pred_df], axis=0, ignore_index=True)

    # base_test에만 있고 sub에 없는 경우 0.5로 채우기
    missing_ids = set(base_test["Test_id"]) - set(sub["Test_id"])
    if missing_ids:
        extra = pd.DataFrame({"Test_id": list(missing_ids), "Label": 0.5})
        sub = pd.concat([sub, extra], axis=0, ignore_index=True)

    sub = sub.set_index("Test_id").loc[base_test["Test_id"]].reset_index()

    # ------------------------------------------------
    # 5. 저장
    # ------------------------------------------------
    sub.to_csv("output/submission.csv", index=False)
    print("[script] output/submission.csv 생성 완료")
    print(f"  총 {len(sub)}개 행")
    print(f"  Label 범위: [{sub['Label'].min():.5f}, {sub['Label'].max():.5f}]")
    print(f"  Label 평균: {sub['Label'].mean():.5f}")


if __name__ == "__main__":
    main()
