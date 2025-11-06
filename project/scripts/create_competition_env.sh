#!/usr/bin/env bash
set -euo pipefail

# 목적: 대회 환경과 동일한 파이썬 패키지 버전으로 가상환경 구성
# 사용:
#   bash project/scripts/create_competition_env.sh

PY=${PYTHON_BIN:-python3.10}

if ! command -v ${PY} >/dev/null 2>&1; then
  echo "[warn] ${PY} 가 없습니다. 대회 환경은 python3.10 기준입니다." >&2
  echo "       macOS에서는 'pyenv' 또는 'brew install python@3.10'을 고려하세요." >&2
fi

${PY} -m venv .venv || python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# 대회 제공 파이썬 패키지 목록 설치 (정확 버전 고정)
pip install -r project/requirements.competition.txt

echo "[done] competition venv ready. Activate with: source .venv/bin/activate"







