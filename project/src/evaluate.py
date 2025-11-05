import numpy as np
from sklearn import metrics


# 한국어 주석: AUC 계산 (이진 분류 ROC AUC)
def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return metrics.roc_auc_score(y_true, y_prob)


# 한국어 주석: Brier 점수 계산 (확률 예측의 평균 제곱 오차)
def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return metrics.brier_score_loss(y_true, y_prob)


# 한국어 주석: ECE(Expected Calibration Error) 계산
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    # 한국어 주석: 구간(bin)을 나누어 예측 확률의 보정 정도를 측정
    y_true = y_true.astype(float)
    y_prob = y_prob.astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    total = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_true = y_true[mask]
        bin_prob = y_prob[mask]
        acc = np.mean((bin_prob >= 0.5).astype(float) == bin_true)
        conf = np.mean(bin_prob)
        ece += (np.sum(mask) / total) * abs(acc - conf)
    return float(ece)


# 한국어 주석: 대회 평가식에 따른 최종 점수 계산 (낮을수록 좋음)
def compute_final_score(auc: float, brier: float, ece: float) -> float:
    return 0.5 * (1.0 - auc) + 0.25 * brier + 0.25 * ece


