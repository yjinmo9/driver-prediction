#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CatBoost 모델을 submit 폴더로 복사하는 스크립트"""

import os
import shutil
from pathlib import Path

def main():
    # 경로 설정
    project_root = Path(__file__).parent.parent
    submit_dir = project_root.parent / "submit" / "model"
    model_dir = project_root / "model"
    
    # submit/model 디렉터리 생성
    submit_dir.mkdir(parents=True, exist_ok=True)
    
    # 복사할 파일 목록
    files_to_copy = [
        "model_A.pkl",
        "model_B.pkl",
        "preproc_A.pkl",
        "preproc_B.pkl",
    ]
    
    print("="*60)
    print("CatBoost 모델 파일 복사")
    print("="*60)
    
    for filename in files_to_copy:
        src = model_dir / filename
        dst = submit_dir / filename
        
        if src.exists():
            shutil.copy2(src, dst)
            print(f"✅ {filename} 복사 완료")
        else:
            print(f"❌ {filename} 파일을 찾을 수 없습니다: {src}")
    
    print("\n[완료] 모델 파일 복사 완료")
    print(f"대상 디렉터리: {submit_dir}")

if __name__ == "__main__":
    main()

