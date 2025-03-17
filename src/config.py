"""
암호화폐 데이터 통합 관리 시스템 설정 파일
"""

import os
from pathlib import Path
from typing import Dict, List

# 기본 디렉토리 설정
DATA_DIR = Path("data")
LIVE_DIR = DATA_DIR / "live"
BACKUP_DIR = DATA_DIR / "backup"
ARCHIVE_DIR = DATA_DIR / "archive"
LONG_TERM_DIR = DATA_DIR / "longterm"

# API 설정
BASE_URL = "https://api.coinalyze.net/v1"
API_REQUEST_INTERVAL = 0.5  # API 요청 사이 대기 시간 (초)
COLLECTION_INTERVAL = 300   # 데이터 수집 간격 (초) - 5분

# API 엔드포인트
ENDPOINTS: Dict[str, str] = {
    "exchanges": "/exchanges",
    "future_markets": "/future-markets",
    "spot_markets": "/spot-markets",
    "open_interest": "/open-interest",
    "funding_rate": "/funding-rate",
    "predicted_funding_rate": "/predicted-funding-rate",
    "open_interest_history": "/open-interest-history",
    "funding_rate_history": "/funding-rate-history",
    "predicted_funding_rate_history": "/predicted-funding-rate-history",
    "liquidation_history": "/liquidation-history",
    "long_short_ratio_history": "/long-short-ratio-history",
    "ohlcv_history": "/ohlcv-history"
}

# API 키 설정 (16개 키를 4개 티어로 구분)
API_KEYS: Dict[str, List[str]] = {
    "tier1": [
        "0d0ba171-4185-4a58-bc02-8c8627cd1f54",
        "e629efa9-68b0-4b6b-b794-fa2d9e379b79",
        "c9779cfd-a85c-4e58-a378-9474174a075e",
        "58a39d7b-84e5-4852-8375-90de055cba18"
    ],
    "tier2": [
        "7b80a59d-f0f9-4a83-81f7-1314dbdd9dc7",
        "2b951698-64e7-4a86-9930-c503d4e29e54",
        "10b24c79-ddd1-4046-94b6-0af0d23b241e",
        "fac040b5-5043-4351-b8f3-09c1d8cfd78f"
    ],
    "tier3": [
        "fe1caf7e-5e27-4f0c-9d14-2a4611db625f",
        "fc9e5080-9607-46e8-b48c-deca57591990",
        "a82fd8a4-873f-4619-9b9f-d8a29373b5b8",
        "6431161c-8815-4d18-846d-55e00863682e"
    ],
    "tier4": [
        "07cc6de0-0d02-41b8-acdc-dd1284bf5730",
        "f2842bed-d43e-4152-a0b5-68d9d9ed30b0",
        "013d0da7-76ea-4699-97d7-6e0f66857939",
        "7906d7bf-b493-42ad-a578-0cd13c6a070c"
    ]
}

# CCXT 설정
CCXT_EXCHANGES: List[str] = ["binance", "bybit", "okx", "kucoin", "bitget"]

# 타임프레임 매핑
TIMEFRAME_MAP: Dict[str, str] = {
    "1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
} 