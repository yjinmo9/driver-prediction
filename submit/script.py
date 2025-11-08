#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CatBoost + HGB 조합 앙상블 모델 제출 스크립트"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import sys
import types

# sklearn 버전 불일치 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================
# 필요한 클래스 정의 (모델 로드 전에 정의 필요)
# =============================================================

class TemperatureScaler:
    """바이너리 확률 Temperature Scaling"""
    def __init__(self, init_T: float = 1.0):
        self.T = float(init_T)

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p) - np.log(1.0 - p)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, y_true: np.ndarray, p_prob: np.ndarray) -> None:
        y = y_true.astype(float)
        z = self._logit(p_prob)
        candidates = np.linspace(0.5, 5.0, 10)
        best_T, best_nll = self.T, np.inf
        for t in candidates:
            p = self._sigmoid(z / t)
            nll = -np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
            if nll < best_nll:
                best_nll, best_T = nll, float(t)
        self.T = best_T

    def transform(self, p_prob: np.ndarray) -> np.ndarray:
        z = self._logit(p_prob)
        p = self._sigmoid(z / max(self.T, 1e-6))
        return np.clip(p, 1e-7, 1 - 1e-7)


class AvgProbaEnsemble:
    """3개의 다른 시드로 학습한 모델의 확률을 평균내는 앙상블"""
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)


class CalibratedWithTemperature:
    """온도 스케일링이 적용된 앙상블"""
    def __init__(self, base_ensemble, temp_scaler):
        self.base_ensemble = base_ensemble
        self.temp_scaler = temp_scaler

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.base_ensemble.predict_proba(X)
        if self.temp_scaler is None:
            return p
        p1 = p[:, 1]
        p1_t = self.temp_scaler.transform(p1)
        p0_t = 1.0 - p1_t
        return np.stack([p0_t, p1_t], axis=1)


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


# =============================================================
# Feature-engineering helpers (copied from training code)
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
        return parts.count(target) / denom

    return series.fillna("").apply(_count_rate)


def _masked_mean_from_csv_series(cond_series: pd.Series, val_series: pd.Series, mask_val: float) -> pd.Series:
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    try:
        cond_arr = cond_df.to_numpy(dtype=float)
    except Exception:
        cond_arr = cond_df.apply(pd.to_numeric, errors="coerce").to_numpy()
    try:
        val_arr = val_df.to_numpy(dtype=float)
    except Exception:
        val_arr = val_df.apply(pd.to_numeric, errors="coerce").to_numpy()

    mask = (cond_arr == mask_val)
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)


def _masked_mean_any_from_csv_series(cond_series: pd.Series, val_series: pd.Series, mask_vals: list[float]) -> pd.Series:
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    val_df = val_series.fillna("").str.split(",", expand=True).replace("", np.nan)
    try:
        cond_arr = cond_df.to_numpy(dtype=float)
    except Exception:
        cond_arr = cond_df.apply(pd.to_numeric, errors="coerce").to_numpy()
    try:
        val_arr = val_df.to_numpy(dtype=float)
    except Exception:
        val_arr = val_df.apply(pd.to_numeric, errors="coerce").to_numpy()

    mask = np.zeros_like(cond_arr, dtype=bool)
    for mv in mask_vals:
        with np.errstate(invalid="ignore"):
            mask |= (cond_arr == mv)
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)


def preprocess_A_v2(train_A: pd.DataFrame) -> pd.DataFrame:
    df = train_A.copy()
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
    df = train_B.copy()
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

    if set(["B4-1", "B4-2"]).issubset(df.columns):
        hard_mean = _masked_mean_any_from_csv_series(df["B4-1"], df["B4-2"], [3, 4, 5, 6])
        easy_mean = _masked_mean_any_from_csv_series(df["B4-1"], df["B4-2"], [1, 2])
        feats["B4_rt_hard_mean"] = hard_mean
        feats["B4_rt_easy_mean"] = easy_mean
        feats["B4_rt_hard_minus_easy"] = hard_mean - easy_mean
        feats["B4_rate_ans1"] = _seq_rate(df["B4-1"], "1")
        feats["B4_rate_ans3"] = _seq_rate(df["B4-1"], "3")
        feats["B4_rate_ans5"] = _seq_rate(df["B4-1"], "5")
        feats["B4_rate_ans35"] = feats["B4_rate_ans3"].fillna(0.0) + feats["B4_rate_ans5"].fillna(0.0)

    for k in ["B6", "B7", "B8"]:
        if k in df.columns:
            feats[f"{k}_acc_rate"] = _seq_rate(df[k], "1")

    if set(["B6_acc_rate", "B7_acc_rate"]).issubset(feats.columns):
        feats["B76_acc_diff"] = feats["B7_acc_rate"] - feats["B6_acc_rate"]

    b9_cols = [c for c in ["B9-1", "B9-2", "B9-3", "B9-4"] if c in df.columns]
    if b9_cols:
        feats["B9_g1to4_mean"] = df[b9_cols].mean(axis=1)
        feats["B9_g1to4_std"] = df[b9_cols].std(axis=1)
    if "B9-5" in df.columns:
        feats["B9_g5"] = df["B9-5"]

    b10_1to4 = [c for c in ["B10-1", "B10-2", "B10-3", "B10-4"] if c in df.columns]
    if b10_1to4:
        feats["B10_g1to4_mean"] = df[b10_1to4].mean(axis=1)
        feats["B10_g1to4_std"] = df[b10_1to4].std(axis=1)
    b10_5to6 = [c for c in ["B10-5", "B10-6"] if c in df.columns]
    if b10_5to6:
        feats["B10_g5to6_mean"] = df[b10_5to6].mean(axis=1)
        feats["B10_g5to6_std"] = df[b10_5to6].std(axis=1)

    if "B10_g1to4_mean" in feats.columns and "B9_g1to4_mean" in feats.columns:
        feats["B10minusB9_g1to4_mean"] = feats["B10_g1to4_mean"] - feats["B9_g1to4_mean"]
    if "B10_g1to4_std" in feats.columns and "B9_g1to4_std" in feats.columns:
        feats["B10minusB9_g1to4_std"] = feats["B10_g1to4_std"] - feats["B9_g1to4_std"]

    base = df.drop(columns=[c for c in df.columns if c in feats.columns], errors="ignore")
    out = pd.concat([base, feats], axis=1)
    return out


def add_rowwise_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols]
    na_count = X.isna().sum(axis=1).astype(np.int32)
    na_ratio = (na_count / (len(feature_cols) + 1e-9)).astype(np.float32)
    df2 = df.copy()
    df2["NA_COUNT"] = na_count
    df2["NA_RATIO"] = na_ratio
    return df2


# =============================================================
# Main inference pipeline
# =============================================================

def main():
    os.makedirs("output", exist_ok=True)

    # 테스트 데이터 로드
    base = pd.read_csv(os.path.join("data", "test.csv"))
    if "Test" not in base.columns:
        raise ValueError("test.csv must contain 'Test' column")

    # Raw features
    A_raw = pd.read_csv(os.path.join("data", "test", "A.csv"))
    B_raw = pd.read_csv(os.path.join("data", "test", "B.csv"))

    # 피처 엔지니어링
    A_feat = preprocess_A_v2(A_raw)
    B_feat = preprocess_B_v2(B_raw)

    # 모델 및 전처리기 로드
    print("[로딩] CatBoost + HGB 조합 앙상블 모델 로드 중...")
    
    # 모델 로드 전에 필요한 클래스를 sys.modules에 등록
    # src.model_utils 모듈을 가짜로 생성
    fake_src = types.ModuleType('src')
    fake_src_model_utils = types.ModuleType('src.model_utils')
    fake_src_model_utils.TemperatureScaler = TemperatureScaler
    fake_src_model_utils.AvgProbaEnsemble = AvgProbaEnsemble
    fake_src_model_utils.CalibratedWithTemperature = CalibratedWithTemperature
    fake_src.model_utils = fake_src_model_utils
    sys.modules['src'] = fake_src
    sys.modules['src.model_utils'] = fake_src_model_utils
    
    # __main__ 모듈에 클래스 등록
    import __main__
    __main__.CatBoostEnsemble = CatBoostEnsemble
    __main__.CombinedEnsemble = CombinedEnsemble
    __main__.TemperatureScaler = TemperatureScaler
    __main__.AvgProbaEnsemble = AvgProbaEnsemble
    __main__.CalibratedWithTemperature = CalibratedWithTemperature
    
    # numpy BitGenerator 호환성 처리 (PCG64 pickle 오류 방지)
    # 평가 서버: numpy==1.26.4, 로컬: numpy 2.x 가능성
    try:
        import pickle
        import numpy.random._pickle
        
        # PCG64 BitGenerator를 pickle에 등록
        try:
            from numpy.random import PCG64
            # PCG64 클래스를 pickle에 등록
            if hasattr(np.random._pickle, '_PCG64_unpickle'):
                pickle.registry[PCG64] = np.random._pickle._PCG64_unpickle
        except (ImportError, AttributeError):
            pass
        
        # joblib의 numpy 호환성 처리
        try:
            import joblib.numpy_pickle
            # joblib이 numpy를 올바르게 인식하도록
            if hasattr(joblib.numpy_pickle, 'NumpyArrayWrapper'):
                pass
        except:
            pass
    except (ImportError, AttributeError):
        # numpy.random._pickle이 없는 경우 (numpy 1.x)
        pass
    
    ensemble_A = joblib.load(os.path.join("model", "model_A.pkl"))
    preproc_A = joblib.load(os.path.join("model", "preproc_A.pkl"))
    ensemble_B = joblib.load(os.path.join("model", "model_B.pkl"))
    preproc_B = joblib.load(os.path.join("model", "preproc_B.pkl"))

    drop_cols_A = ["Test_id", "Label"]
    drop_cols_B = ["Test_id", "Label"]

    preds_list = []

    # A 모델 예측
    A_idx = base[base["Test"] == "A"].copy()
    if len(A_idx):
        print(f"[예측] A 모델 예측 중... (n={len(A_idx)})")
        df = A_idx.merge(A_feat, on="Test_id", how="left", validate="1:1")
        if 'Test_x' in df.columns or 'Test_y' in df.columns:
            if 'Test_x' in df.columns:
                df.rename(columns={'Test_x': 'Test'}, inplace=True)
            df.drop(columns=[c for c in ['Test_y'] if c in df.columns], inplace=True, errors='ignore')
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 피처 준비
        feature_cols = [c for c in df.columns if c not in drop_cols_A]
        df = add_rowwise_features(df, feature_cols)
        X = df.drop(columns=drop_cols_A, errors="ignore")
        
        # 전처리 및 예측
        X_t = preproc_A.transform(X)
        pred = ensemble_A.predict_proba(X_t)[:, 1]
        preds_list.append(pd.DataFrame({"Test_id": df["Test_id"], "Label": pred}))

    # B 모델 예측
    B_idx = base[base["Test"] == "B"].copy()
    if len(B_idx):
        print(f"[예측] B 모델 예측 중... (n={len(B_idx)})")
        df = B_idx.merge(B_feat, on="Test_id", how="left", validate="1:1")
        if 'Test_x' in df.columns or 'Test_y' in df.columns:
            if 'Test_x' in df.columns:
                df.rename(columns={'Test_x': 'Test'}, inplace=True)
            df.drop(columns=[c for c in ['Test_y'] if c in df.columns], inplace=True, errors='ignore')
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 피처 준비
        feature_cols = [c for c in df.columns if c not in drop_cols_B]
        df = add_rowwise_features(df, feature_cols)
        X = df.drop(columns=drop_cols_B, errors="ignore")
        
        # 전처리 및 예측
        X_t = preproc_B.transform(X)
        pred = ensemble_B.predict_proba(X_t)[:, 1]
        preds_list.append(pd.DataFrame({"Test_id": df["Test_id"], "Label": pred}))

    # 결과 병합
    if preds_list:
        sub = pd.concat(preds_list, axis=0, ignore_index=True)
    else:
        sub = pd.DataFrame({"Test_id": base["Test_id"], "Label": 0.5})

    # Sample submission과 병합 (컬럼 순서 맞추기)
    try:
        sample = pd.read_csv(os.path.join("data", "sample_submission.csv"))
        sub = sample.merge(sub, on="Test_id", how="left", suffixes=("", "_pred"))
        if "Label_pred" in sub.columns:
            sub["Label"] = sub["Label_pred"]
        elif "Label_y" in sub.columns:
            sub["Label"] = sub["Label_y"]
        elif "Label_x" in sub.columns and "Label" not in sub.columns:
            sub.rename(columns={"Label_x": "Label"}, inplace=True)
        if "Label" not in sub.columns:
            sub["Label"] = 0.5
        sub["Label"] = sub["Label"].astype(float).fillna(0.5)
        sub = sub[["Test_id", "Label"]]
    except Exception:
        sub = sub[["Test_id", "Label"]]

    # 결과 저장
    sub.to_csv(os.path.join("output", "submission.csv"), index=False)
    print(f"✅ output/submission.csv 저장 완료 (rows={len(sub)})")
    print(f"   Label 범위: [{sub['Label'].min():.5f}, {sub['Label'].max():.5f}]")
    print(f"   Label 평균: {sub['Label'].mean():.5f}")


if __name__ == "__main__":
    main()

