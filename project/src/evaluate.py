import numpy as np
from sklearn import metrics


# 한국어 주석: AUC 계산 (이진 분류 ROC AUC)
def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return metrics.roc_auc_score(y_true, y_prob)


# 한국어 주석: Brier 점수 계산 (확률 예측의 평균 제곱 오차)
def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return metrics.brier_score_loss(y_true, y_prob)


# 한국어 주석: ECE(Expected Calibration Error) 계산 (표준 정의)
# 각 bin에서 acc=mean(y_true), conf=mean(y_prob)
# 가중합 sum_n (n/N) * |acc - conf|
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
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
        acc = np.mean(bin_true)
        conf = np.mean(bin_prob)
        ece += (np.sum(mask) / total) * abs(acc - conf)
    return float(ece)


# 한국어 주석: 대회 평가식에 따른 최종 점수 계산 (낮을수록 좋음)
def compute_final_score(auc: float, brier: float, ece: float) -> float:
    return 0.5 * (1.0 - auc) + 0.25 * brier + 0.25 * ece


# 한국어 주석: 테스트 데이터의 클래스 분포가 다를 경우를 위한 리밸런싱 (Bayes' theorem)
def rebalance_probabilities(
    proba: np.ndarray,
    train_prior: float,
    test_prior: float = None,
    method: str = "bayes"
) -> np.ndarray:
    """
    테스트 데이터의 클래스 분포가 학습 데이터와 다를 경우 확률 조정
    
    Args:
        proba: 모델이 예측한 확률 (0~1)
        train_prior: 학습 데이터의 positive 클래스 비율
        test_prior: 테스트 데이터의 positive 클래스 비율 (None이면 train_prior 사용)
        method: 조정 방법 ("bayes" 또는 "linear")
    
    Returns:
        조정된 확률
    """
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    
    if test_prior is None:
        test_prior = train_prior
    
    if method == "bayes":
        # Bayes' theorem을 사용한 조정
        # P(y=1|x) = P(x|y=1) * P(y=1) / P(x)
        # odds = P(y=1|x) / P(y=0|x) = P(x|y=1) / P(x|y=0) * P(y=1) / P(y=0)
        # odds_new = odds * (P_test(y=1) / P_test(y=0)) / (P_train(y=1) / P_train(y=0))
        
        train_odds = train_prior / (1 - train_prior)
        test_odds = test_prior / (1 - test_prior)
        
        odds_ratio = test_odds / train_odds
        odds = proba / (1 - proba)
        adjusted_odds = odds * odds_ratio
        adjusted_proba = adjusted_odds / (1 + adjusted_odds)
        
    elif method == "linear":
        # 선형 조정: 단순히 비율만큼 조정
        ratio = test_prior / train_prior
        adjusted_proba = proba * ratio
        adjusted_proba = np.clip(adjusted_proba, 0, 1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return np.clip(adjusted_proba, 1e-7, 1 - 1e-7)


