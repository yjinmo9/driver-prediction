#!/usr/bin/env bash
set -euo pipefail

# 목적: macOS에서 pyenv로 Python 3.10.x를 설치/적용하고, 대회용 가상환경 재구성
# 사용법:
#   bash project/scripts/use_pyenv_python310.sh

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYVER="3.10.13"

if ! command -v brew >/dev/null 2>&1; then
  echo "[error] Homebrew가 없습니다. https://brew.sh/ 참고하여 설치 후 다시 시도하세요." >&2
  exit 1
fi

if ! command -v pyenv >/dev/null 2>&1; then
  echo "[info] pyenv가 없어 설치합니다..."
  brew update
  brew install pyenv
  echo "[note] 다음을 쉘 설정(~/.zshrc 등)에 추가 후 새 터미널을 여세요:"
  echo '  export PYENV_ROOT="$HOME/.pyenv"'
  echo '  command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"'
  echo '  eval "$(pyenv init -)"'
  echo "[note] 현재 셸 세션에 임시 반영합니다."
  export PYENV_ROOT="$HOME/.pyenv"
  command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"
fi

# pyenv 초기화 (스크립트 재실행 시에도 안전)
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

if ! pyenv versions --bare | grep -q "^${PYVER}$"; then
  echo "[info] Installing Python ${PYVER} via pyenv..."
  # macOS 빌드에 필요한 라이브러리 권장 패키지
  brew install openssl readline sqlite3 xz zlib || true
  CFLAGS="-I$(brew --prefix openssl)/include" \
  LDFLAGS="-L$(brew --prefix openssl)/lib" \
  pyenv install "${PYVER}"
fi

echo "[info] Setting local python to ${PYVER}"
pyenv local "${PYVER}"
echo "[info] Python version: $(python -V)"

echo "[info] Reset virtualenv (.venv)"
rm -rf .venv || true

echo "[info] Create competition env with pinned packages"
bash project/scripts/create_competition_env.sh

echo "[done] pyenv 설정 및 대회용 가상환경 구성이 완료되었습니다."
echo "       가상환경 활성화:  source .venv/bin/activate"







