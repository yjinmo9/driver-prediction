import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import DATA_DIR, TRAIN_DIR, TEST_DIR


# 한국어 주석: 숫자형 다운캐스팅을 통해 메모리 사용량을 줄이는 함수
def optimize_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # 한국어 주석: 각 숫자형 컬럼에 대해 가능한 가장 작은 정밀도의 dtype으로 다운캐스팅
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col]
        if pd.api.types.is_float_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast="float")
        elif pd.api.types.is_integer_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast="integer")
    return df


# 한국어 주석: 범주형 추정이 가능한 컬럼을 category로 캐스팅하여 메모리 절감
def optimize_categorical_dtypes(df: pd.DataFrame, max_unique_ratio: float = 0.2) -> pd.DataFrame:
    # 한국어 주석: 고유값 비율이 낮은 object 컬럼은 category로 변경 (문자열 메모리 절감)
    n_rows = max(len(df), 1)
    for col in df.select_dtypes(include=["object"]).columns:
        num_unique = df[col].nunique(dropna=False)
        if (num_unique / n_rows) <= max_unique_ratio:
            df[col] = df[col].astype("category")
    return df


# 한국어 주석: 공통 다운캐스팅/최적화 래퍼
def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    df = optimize_numeric_dtypes(df)
    df = optimize_categorical_dtypes(df)
    return df


# 한국어 주석: Test 열에 따라 A/B 상세 데이터와 병합하는 함수 (train 전용)
def load_and_merge_train(train_csv_path: str) -> pd.DataFrame:
    # 한국어 주석: 상위 목록 파일(train.csv) 로드
    base = pd.read_csv(train_csv_path)

    # 한국어 주석: A/B 상세 데이터 로드
    a_path = os.path.join(TRAIN_DIR, "A.csv")
    b_path = os.path.join(TRAIN_DIR, "B.csv")
    a_df = pd.read_csv(a_path)
    b_df = pd.read_csv(b_path)

    # 한국어 주석: A/B 각각 병합 -> 스키마 통일 -> 세로 결합
    a_base = base[base["Test"] == "A"].merge(a_df, on="Test_id", how="left")
    b_base = base[base["Test"] == "B"].merge(b_df, on="Test_id", how="left")

    # 한국어 주석: 스키마 통일을 위해 컬럼 합집합 생성
    all_cols: List[str] = sorted(list(set(a_base.columns) | set(b_base.columns)))
    a_base = a_base.reindex(columns=all_cols)
    b_base = b_base.reindex(columns=all_cols)

    merged = pd.concat([a_base, b_base], axis=0, ignore_index=True)
    merged = optimize_memory(merged)
    return merged


# 한국어 주석: Test 열에 따라 A/B 상세 데이터와 병합하는 함수 (test 전용)
def load_and_merge_test(test_csv_path: str) -> pd.DataFrame:
    # 한국어 주석: 상위 목록 파일(test.csv) 로드
    base = pd.read_csv(test_csv_path)

    # 한국어 주석: A/B 상세 데이터 로드
    a_path = os.path.join(TEST_DIR, "A.csv")
    b_path = os.path.join(TEST_DIR, "B.csv")
    a_df = pd.read_csv(a_path)
    b_df = pd.read_csv(b_path)

    # 한국어 주석: A/B 각각 병합 -> 스키마 통일 -> 세로 결합
    a_base = base[base["Test"] == "A"].merge(a_df, on="Test_id", how="left")
    b_base = base[base["Test"] == "B"].merge(b_df, on="Test_id", how="left")

    all_cols: List[str] = sorted(list(set(a_base.columns) | set(b_base.columns)))
    a_base = a_base.reindex(columns=all_cols)
    b_base = b_base.reindex(columns=all_cols)

    merged = pd.concat([a_base, b_base], axis=0, ignore_index=True)
    merged = optimize_memory(merged)
    return merged


# 한국어 주석: dtype 정보를 직렬화 가능한 문자열 사전으로 변환
def extract_dtypes(df: pd.DataFrame, columns: List[str]) -> Dict[str, str]:
    # 한국어 주석: 각 컬럼의 dtype 이름을 문자열로 저장 (재현성)
    dtype_map: Dict[str, str] = {}
    for c in columns:
        if c in df.columns:
            dtype_map[c] = str(df[c].dtype)
    return dtype_map


# 한국어 주석: 인덱스 파일(train.csv, test.csv) 로드 (참고 코드 방식)
def read_index_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 한국어 주석: train.csv와 test.csv를 각각 로드
    train_idx = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_idx = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_idx, test_idx


# 한국어 주석: A/B 상세 데이터 파일 로드 (참고 코드 방식)
def read_feature_files(split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 한국어 주석: train 또는 test 디렉터리에서 A.csv, B.csv 로드
    A_df = pd.read_csv(os.path.join(DATA_DIR, split, "A.csv"))
    B_df = pd.read_csv(os.path.join(DATA_DIR, split, "B.csv"))
    return A_df, B_df


