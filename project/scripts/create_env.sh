#!/usr/bin/env bash
set -euo pipefail

# 목적: 파이썬 가상환경 생성 및 필수 패키지 설치
# 사용법:
#   bash project/scripts/create_env.sh

PY=${PYTHON_BIN:-python3}

${PY} -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# macOS 등에서 matplotlib 소스 빌드 회피: 바이너리 휠로 선설치
pip install --only-binary=:all: "matplotlib==3.10.7" "seaborn==0.13.2" || true

# 프로젝트 기본 요구사항 설치
pip install -r project/requirements.txt

# 선택: EDA/그래프 용도 추가 패키지
if [ -f project/requirements.eda.txt ]; then
  pip install -r project/requirements.eda.txt
fi

echo "[done] virtualenv ready. Activate with: source .venv/bin/activate"





