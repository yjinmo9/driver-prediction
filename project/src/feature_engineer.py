import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .config import DEFAULT_DROP_COLUMNS


# 한국어 주석: 제거할 기본 컬럼 목록을 반환 (존재하지 않으면 무시됨)
def get_drop_columns(df: pd.DataFrame) -> List[str]:
    # 한국어 주석: DEFAULT_DROP_COLUMNS에 정의된 컬럼들 중 실제 존재하는 것만 반환
    return [c for c in DEFAULT_DROP_COLUMNS if c in df.columns]


# 한국어 주석: 전처리 대상 피처 컬럼 목록을 정의 (라벨/식별자 제외)
def get_feature_columns(df: pd.DataFrame, label_col: str = "Label") -> List[str]:
    drop_cols = set(get_drop_columns(df) + [label_col])
    return [c for c in df.columns if c not in drop_cols]


# 한국어 주석: 수치/범주형 컬럼을 자동 분리
def split_numeric_categorical(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


# 한국어 주석: 학습 시점의 컬럼 정보를 기반으로 전처리 파이프라인을 생성
def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    numeric_cols, categorical_cols = split_numeric_categorical(df, feature_cols)

    # 한국어 주석: 수치형 - 중앙값 대치
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # 한국어 주석: 범주형 - 최빈값 대치 후 Ordinal 인코딩 (메모리 효율, 미지의 값은 -1 처리)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordenc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,  # 한국어 주석: sparse matrix를 dense로 반환 (참고 코드 방식)
    )
    return preprocessor


# 한국어 주석: 행 단위 피처 생성 (결측치 개수/비율) - 참고 코드 방식
def add_rowwise_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # 한국어 주석: 각 행별로 결측치가 몇 개나 있는지 계산
    X = df[feature_cols]
    na_count = X.isna().sum(axis=1).astype(np.int32)
    na_ratio = (na_count / (len(feature_cols) + 1e-9)).astype(np.float32)
    
    # 한국어 주석: 원본 데이터프레임에 새로운 컬럼 추가
    df2 = df.copy()
    df2["NA_COUNT"] = na_count
    df2["NA_RATIO"] = na_ratio
    return df2


# 한국어 주석: Age에서 숫자와 접미사(a/b) 분리
# EDA 결과: a=해당 연령대 초반, b=해당 연령대 후반 (예: 20a=20대 초반, 20b=20대 후반)
def split_age(s) -> Tuple[Optional[float], Optional[str]]:
    if pd.isna(s):
        return (None, None)
    m = re.match(r'^(\d+)([A-Za-z]?)$', str(s).strip())
    if not m:
        return (None, None)
    num = int(m.group(1))
    suf = m.group(2).lower() if m.group(2) else None
    return (num, suf)


# 한국어 주석: TestDate를 날짜로 파싱 - EDA에서 사용한 로직
def parse_testdate(s) -> pd.Timestamp:
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    # 6자리(YYYYMM) 또는 8자리(YYYYMMDD) 모두 허용
    if re.match(r'^\d{6}$', s):
        return pd.to_datetime(s+'01', format='%Y%m%d', errors='coerce')
    if re.match(r'^\d{8}$', s):
        return pd.to_datetime(s, format='%Y%m%d', errors='coerce')
    # 하이픈 포함 포맷도 허용
    try:
        return pd.to_datetime(s, errors='coerce')
    except Exception:
        return pd.NaT


# 한국어 주석: Age 분해 및 TestDate 파싱 피처 추가
def add_age_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    
    # Age 분해 (a=초반, b=후반)
    if 'Age' in df2.columns:
        age_parts = df2['Age'].apply(lambda x: pd.Series(split_age(x), index=['Age_num', 'Age_suffix']))
        df2['Age_num'] = age_parts['Age_num']
        df2['Age_suffix'] = age_parts['Age_suffix']
        
        # Age_suffix를 숫자로 인코딩 (a=0 초반, b=1 후반, None=0.5)
        df2['Age_suffix_encoded'] = df2['Age_suffix'].map({'a': 0, 'b': 1}).fillna(0.5)
        
        # 연령대 중간값 계산 (20a=22.5세, 20b=27.5세, 20=25세)
        def calc_age_mid(row):
            age_num = row['Age_num']
            suffix = row['Age_suffix']
            if pd.isna(age_num):
                return np.nan
            if suffix == 'a':
                return age_num + 2.5  # 초반 중간값
            elif suffix == 'b':
                return age_num + 7.5  # 후반 중간값
            else:
                return age_num + 5.0  # 해당 연령대 중간값
        df2['Age_mid'] = df2.apply(calc_age_mid, axis=1)
        
        # 10년 단위 연령대 (20대=2, 30대=3, ...)
        df2['Age_decade'] = (df2['Age_num'] // 10).astype(float)
    
    # TestDate 파싱
    if 'TestDate' in df2.columns:
        df2['TestDate_parsed'] = df2['TestDate'].map(parse_testdate)
        # 날짜 기반 피처 (연, 월, 일)
        df2['TestDate_year'] = df2['TestDate_parsed'].dt.year
        df2['TestDate_month'] = df2['TestDate_parsed'].dt.month
        df2['TestDate_day'] = df2['TestDate_parsed'].dt.day
        df2['TestDate_weekday'] = df2['TestDate_parsed'].dt.weekday
    
    return df2


# 한국어 주석: PrimaryKey 기준 시간순 세션 인덱스 생성
def add_session_features(df: pd.DataFrame, key_col: str = 'PrimaryKey', date_col: str = 'TestDate_parsed') -> pd.DataFrame:
    df2 = df.copy()
    if key_col not in df2.columns or date_col not in df2.columns:
        return df2
    
    # 사람별 시간순 정렬 후 순위(세션 인덱스) 부여
    df2 = df2.sort_values([key_col, date_col])
    df2['session_idx'] = df2.groupby(key_col).cumcount() + 1
    df2['total_sessions'] = df2.groupby(key_col)[key_col].transform('count')
    
    # 이전 측정까지의 시간 간격 (일 단위)
    df2['days_since_last'] = df2.groupby(key_col)[date_col].diff().dt.days
    df2['days_since_first'] = (df2[date_col] - df2.groupby(key_col)[date_col].transform('min')).dt.days
    
    # 나이 변화 관련 피처 (한 사람이 시간에 따라 나이를 먹는 패턴)
    if 'Age_num' in df2.columns:
        # 이전 측정 대비 Age_num 변화
        df2['age_num_change'] = df2.groupby(key_col)['Age_num'].diff()
        df2['age_num_change_rate'] = df2['age_num_change'] / (df2['days_since_last'] + 1e-9)  # 일당 변화율
        
        # 연령대 전환 여부 (예: 20대→30대)
        df2['decade'] = df2['Age_num'] // 10
        df2['decade_change'] = df2.groupby(key_col)['decade'].diff()
        df2['decade_changed'] = (df2['decade_change'] != 0).astype(int).fillna(0)
        
        # 접미사 변화 여부 (a→b, b→a 등)
        if 'Age_suffix' in df2.columns:
            df2['suffix_changed'] = (df2.groupby(key_col)['Age_suffix'].shift(1) != df2['Age_suffix']).astype(int)
            df2['suffix_changed'] = df2['suffix_changed'].fillna(0)
        
        # 첫 측정 대비 나이 변화
        df2['age_num_from_first'] = df2['Age_num'] - df2.groupby(key_col)['Age_num'].transform('first')
    
    return df2


# 한국어 주석: 문자열 리스트 컬럼 파싱 (예: A2-2 "1,1,3,2,2,1,...")
def parse_list_column(s, stats: bool = True) -> dict:
    if pd.isna(s):
        return {'list_len': 0, 'list_nunique': 0, 'list_mean': np.nan, 'list_std': np.nan}
    try:
        parts = str(s).split(',')
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return {'list_len': 0, 'list_nunique': 0, 'list_mean': np.nan, 'list_std': np.nan}
        # 숫자로 변환 가능한지 확인
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except:
                pass
        if not nums:
            return {'list_len': len(parts), 'list_nunique': len(set(parts)), 'list_mean': np.nan, 'list_std': np.nan}
        return {
            'list_len': len(nums),
            'list_nunique': len(set(nums)),
            'list_mean': float(np.mean(nums)),
            'list_std': float(np.std(nums)) if len(nums) > 1 else 0.0,
        }
    except:
        return {'list_len': 0, 'list_nunique': 0, 'list_mean': np.nan, 'list_std': np.nan}


# 한국어 주석: 문자열 리스트 컬럼에서 피처 추출
def add_list_features(df: pd.DataFrame, list_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df2 = df.copy()
    
    if list_cols is None:
        # 자동 탐지: 쉼표 포함 문자열 컬럼
        list_cols = []
        for c in df2.select_dtypes(include=['object']).columns:
            sample = df2[c].dropna().head(100)
            if len(sample) > 0 and sample.astype(str).str.contains(',').any():
                list_cols.append(c)
    
    for col in list_cols:
        if col not in df2.columns:
            continue
        parsed = df2[col].apply(lambda x: pd.Series(parse_list_column(x)))
        for stat in ['list_len', 'list_nunique', 'list_mean', 'list_std']:
            df2[f'{col}_{stat}'] = parsed[stat]
    
    return df2


# 한국어 주석: PrimaryKey 기준 집계 피처 (사람별 통계)
def add_person_aggregate_features(df: pd.DataFrame, key_col: str = 'PrimaryKey', 
                                   numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df2 = df.copy()
    if key_col not in df2.columns:
        return df2
    
    if numeric_cols is None:
        # 자동 선택: 숫자형 중 식별자 제외
        drop_set = set(DEFAULT_DROP_COLUMNS + [key_col, 'Age_num', 'session_idx'])
        numeric_cols = [c for c in df2.select_dtypes(include=[np.number]).columns 
                        if c not in drop_set]
    
    # 사람별 통계 (평균, 중앙값, 표준편차, 최소, 최대)
    for col in numeric_cols[:20]:  # 상위 20개만 (메모리 고려)
        if col not in df2.columns:
            continue
        agg_stats = df2.groupby(key_col)[col].agg(['mean', 'median', 'std', 'min', 'max'])
        for stat in ['mean', 'median', 'std', 'min', 'max']:
            df2[f'{col}_person_{stat}'] = df2[key_col].map(agg_stats[stat])
    
    return df2


# 한국어 주석: 반응시간 관련 통계 피처 (컬럼명 패턴 기반)
def add_response_time_features(df: pd.DataFrame, patterns: List[str] = ['response', 'time', 'rt']) -> pd.DataFrame:
    df2 = df.copy()
    
    # 패턴 매칭
    regex = re.compile('|'.join(patterns), flags=re.IGNORECASE)
    rt_cols = [c for c in df2.columns if regex.search(str(c)) and pd.api.types.is_numeric_dtype(df2[c])]
    
    if rt_cols:
        # 반응시간 평균, 중앙값, 최소, 최대
        df2['rt_mean'] = df2[rt_cols].mean(axis=1)
        df2['rt_median'] = df2[rt_cols].median(axis=1)
        df2['rt_min'] = df2[rt_cols].min(axis=1)
        df2['rt_max'] = df2[rt_cols].max(axis=1)
        df2['rt_std'] = df2[rt_cols].std(axis=1)
    
    return df2


# 한국어 주석: 통합 피처 엔지니어링 함수
def engineer_features(df: pd.DataFrame, 
                     add_age_time: bool = True,
                     add_session: bool = True,
                     add_list: bool = True,
                     add_person_agg: bool = True,
                     add_rt: bool = True) -> pd.DataFrame:
    """EDA 결과를 바탕으로 통합 피처 엔지니어링"""
    df2 = df.copy()
    
    if add_age_time:
        df2 = add_age_time_features(df2)
    
    if add_session and 'PrimaryKey' in df2.columns and 'TestDate_parsed' in df2.columns:
        df2 = add_session_features(df2)
    
    if add_list:
        df2 = add_list_features(df2)
    
    if add_person_agg and 'PrimaryKey' in df2.columns:
        df2 = add_person_aggregate_features(df2)
    
    if add_rt:
        df2 = add_response_time_features(df2)
    
    return df2



# =========================
# v2 Feature Engineering
# =========================

from tqdm.auto import tqdm  # type: ignore

# tqdm for apply (노트북/스크립트 모두에서 무해)
try:
    tqdm.pandas()  # noqa: E701  # progress_apply 활성화
except Exception:
    pass


def _seq_mean(series: pd.Series) -> pd.Series:
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(str(x), sep=",").mean() if str(x) else np.nan
    )


def _seq_std(series: pd.Series) -> pd.Series:
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(str(x), sep=",").std() if str(x) else np.nan
    )


def _seq_rate(series: pd.Series, target: str = "1") -> pd.Series:
    # 시퀀스에서 target 비율 계산
    def _count_rate(x: str) -> float:
        x = str(x)
        if not x:
            return np.nan
        parts = x.split(",")
        denom = len(parts) if len(parts) > 0 else np.nan
        if denom == 0 or np.isnan(denom):
            return np.nan
        return parts.count(target) / denom

    return series.fillna("").progress_apply(_count_rate)


def _masked_mean_from_csv_series(cond_series: pd.Series, val_series: pd.Series, mask_val: float) -> pd.Series:
    """조건 시퀀스(cond_series)에서 mask_val 위치의 val_series 평균"""
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


def preprocess_A_v2(train_A: pd.DataFrame) -> pd.DataFrame:
    """사용자 정의 A 데이터 피처 엔지니어링(v2 1차안).
    - 쉼표 시퀀스 기반 요약(비율/평균/표준편차)
    - 과제별 주요 파생
    - 시퀀스 원본은 제거
    """
    df = train_A.copy()

    # 0) 결측치가 너무 많은 행은 제거 (요구사항에 맞춤)
    df = df.dropna().reset_index(drop=True)

    # 1) 불필요한 열 제거 (존재할 때만)
    drop_cols = [
        "A1-1", "A1-2",
        "A3-1", "A3-2", "A3-3", "A3-4", "A3-5",
        "A4-1", "A4-2", "A4-3",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    feats = pd.DataFrame(index=df.index)

    # ---- A1 ----
    if "A1-3" in df.columns:
        feats["A1_resp_rate"] = _seq_rate(df["A1-3"], "0")  # 0=반응함
    if "A1-4" in df.columns:
        feats["A1_rt_mean"] = _seq_mean(df["A1-4"])  
        feats["A1_rt_std"] = _seq_std(df["A1-4"])   

    # ---- A2 ----
    if "A2-3" in df.columns:
        feats["A2_resp_rate"] = _seq_rate(df["A2-3"], "0")
    if "A2-4" in df.columns:
        feats["A2_rt_mean"] = _seq_mean(df["A2-4"])  
        feats["A2_rt_std"] = _seq_std(df["A2-4"])   

    # ---- A3 ----
    if "A3-6" in df.columns:
        feats["A3_resp_rate"] = _seq_rate(df["A3-6"], "0")
    if "A3-7" in df.columns:
        feats["A3_rt_mean"] = _seq_mean(df["A3-7"])  
        feats["A3_rt_std"] = _seq_std(df["A3-7"])   

    # ---- A4 ----
    if "A4-4" in df.columns:
        feats["A4_resp_rate"] = _seq_rate(df["A4-4"], "0")
    if "A4-3" in df.columns:
        feats["A4_acc_rate"] = _seq_rate(df["A4-3"], "1")
    if "A4-5" in df.columns:
        feats["A4_rt_mean"] = _seq_mean(df["A4-5"])  
        feats["A4_rt_std"] = _seq_std(df["A4-5"])   
    # stroop diff: (cond=2) - (cond=1) on A4-5
    if set(["A4-1", "A4-5"]).issubset(df.columns):
        feats["A4_stroop_diff"] = _masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 2) - \
                                    _masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 1)

    # ---- A5 ----
    if "A5-3" in df.columns:
        feats["A5_resp_rate"] = _seq_rate(df["A5-3"], "0")
    if "A5-2" in df.columns:
        feats["A5_acc_rate"] = _seq_rate(df["A5-2"], "1")

    # ---- A6~A9 (집계형 수치 그대로 사용) ----
    for col in [
        "A6-1", "A7-1", "A8-1", "A8-2",
        "A9-1", "A9-2", "A9-3", "A9-4", "A9-5",
    ]:
        if col in df.columns:
            feats[col] = df[col]

    # 시퀀스 원본 제거 후 결합
    seq_cols = [c for c in df.columns if "-" in c and c not in feats.columns]
    out = pd.concat([df.drop(columns=seq_cols, errors="ignore"), feats], axis=1)
    # 알림
    print("✅ A 피처 엔지니어링(v2) 완료:", out.shape)
    return out


def preprocess_B_v2(train_B: pd.DataFrame) -> pd.DataFrame:
    """사용자 정의 B 데이터 피처 엔지니어링(v2 1차안)."""
    df = train_B.copy()
    df = df.dropna().reset_index(drop=True)

    feats = pd.DataFrame(index=df.index)

    # ---- B1 ----
    if "B1-1" in df.columns:
        feats["B1_acc_rate"] = _seq_rate(df["B1-1"], "1")
    if "B1-2" in df.columns:
        feats["B1_rt_mean"] = _seq_mean(df["B1-2"])  
        feats["B1_rt_std"] = _seq_std(df["B1-2"])   

    # ---- B2 ----
    if "B2-1" in df.columns:
        feats["B2_acc_rate"] = _seq_rate(df["B2-1"], "1")
    if "B2-2" in df.columns:
        feats["B2_rt_mean"] = _seq_mean(df["B2-2"])  
        feats["B2_rt_std"] = _seq_std(df["B2-2"])   

    # ---- B3~B5 ----
    for k in ["B3", "B4", "B5"]:
        acc_col, rt_col = f"{k}-1", f"{k}-2"
        if acc_col in df.columns:
            feats[f"{k}_acc_rate"] = _seq_rate(df[acc_col], "1")
        if rt_col in df.columns:
            feats[f"{k}_rt_mean"] = _seq_mean(df[rt_col])
            feats[f"{k}_rt_std"] = _seq_std(df[rt_col])

    # ---- B6~B8 ----
    for k in ["B6", "B7", "B8"]:
        if k in df.columns:
            feats[f"{k}_acc_rate"] = _seq_rate(df[k], "1")

    # ---- B9~B10 (집계형 count 그대로 사용) ----
    for col in [
        "B9-1", "B9-2", "B9-3", "B9-4", "B9-5",
        "B10-1", "B10-2", "B10-3", "B10-4", "B10-5", "B10-6",
    ]:
        if col in df.columns:
            feats[col] = df[col]

    out = pd.concat([
        df.drop(columns=[c for c in df.columns if "-" in c and c not in feats.columns], errors="ignore"),
        feats,
    ], axis=1)
    print("✅ B 피처 엔지니어링(v2) 완료:", out.shape)
    return out
