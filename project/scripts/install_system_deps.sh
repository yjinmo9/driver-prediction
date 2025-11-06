#!/usr/bin/env bash
set -euo pipefail

# 목적: Debian/Ubuntu 계열에서 EDA/ML 파이프라인 구동에 필요한 시스템 패키지 설치
# 사용법:
#   bash project/scripts/install_system_deps.sh

if ! command -v apt-get >/dev/null 2>&1; then
  echo "[info] apt-get 이 없는 환경입니다 (macOS 등). 이 스크립트는 Debian/Ubuntu 전용입니다." >&2
  echo "[hint] macOS에서는 'brew install poppler ffmpeg qpdf jpeg libpng openjpeg' 등을 사용하세요." >&2
  exit 0
fi

sudo apt-get update -y
sudo apt-get install -y \
  python3.10 python3-pip python3.10-distutils \
  build-essential gfortran libffi-dev \
  libblas3 liblapack3 libomp-dev \
  libpng-dev libjpeg-dev libopenjp2-7 \
  poppler-utils ffmpeg unzip p7zip-full \
  pdftk-java qpdf tzdata default-jre git

echo "[done] system dependencies installed"










