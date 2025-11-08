#!/usr/bin/env python3
import os
import sys
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"

# src 모듈을 찾을 수 있도록 경로 추가
sys.path.insert(0, str(PROJECT_ROOT))


def export_bundle(which: str) -> None:
    model_path = MODEL_DIR / f"model_{which}.pkl"
    preproc_path = MODEL_DIR / f"preproc_{which}.pkl"
    bundle_path = MODEL_DIR / f"bundle_{which}.pkl"

    if not model_path.exists() or not preproc_path.exists():
        raise FileNotFoundError(f"Required files for {which} not found")

    model = joblib.load(model_path)
    preproc = joblib.load(preproc_path)

    base = getattr(model, "base_ensemble", None)
    temp_scaler = getattr(model, "temp_scaler", None)
    temperature = float(getattr(temp_scaler, "T", 1.0))

    if base is None or not hasattr(base, "models"):
        raise ValueError("Unexpected model structure: base_ensemble missing")

    portable = {
        "preproc": preproc,
        "models": list(base.models),
        "temperature": temperature,
    }

    joblib.dump(portable, bundle_path)
    print(f"Saved bundle -> {bundle_path}")


def main():
    export_bundle("A")
    export_bundle("B")


if __name__ == "__main__":
    main()
