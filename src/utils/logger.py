"""
로깅 설정 모듈
"""

import logging
from pathlib import Path

def setup_logger():
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "crypto_data_manager.log", encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__)

# 전역 로거 인스턴스 생성
logger = setup_logger() 