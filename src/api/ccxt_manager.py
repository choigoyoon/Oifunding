"""
CCXT 거래소 관리 모듈
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import ccxt

from ..config import CCXT_EXCHANGES, TIMEFRAME_MAP, DATA_DIR
from ..utils.logger import logger

class CcxtManager:
    """CCXT 거래소 관리 및 심볼 매핑 클래스"""
    def __init__(self):
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.symbol_mappings: Dict[str, Dict[str, str]] = {}  # base_symbol -> {exchange_id -> ccxt_symbol}
        self.exchange_priority = ["binance", "bybit", "okx", "kucoin", "bitget"]
        self.supported_symbols: Set[str] = set()  # CCXT에서 지원하는 심볼 목록
        self.timeframe_map = TIMEFRAME_MAP
        self.init_exchanges()
        
    def init_exchanges(self):
        """CCXT 거래소 초기화 (API 키 없이)"""
        for exchange_id in CCXT_EXCHANGES:
            try:
                if exchange_id in ccxt.exchanges:
                    exchange_class = getattr(ccxt, exchange_id)
                    self.exchanges[exchange_id] = exchange_class({
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'future'  # 선물 거래 기본 설정
                        }
                    })
                    logger.info(f"CCXT {exchange_id} 거래소 초기화 성공")
            except Exception as e:
                logger.error(f"CCXT {exchange_id} 거래소 초기화 실패: {str(e)}")
    
    def load_all_markets(self):
        """모든 거래소의 시장 정보 로드"""
        for exchange_id, exchange in self.exchanges.items():
            try:
                exchange.load_markets()
                logger.info(f"{exchange_id} 시장 정보 로드 성공: {len(exchange.symbols)}개 심볼")
            except Exception as e:
                logger.error(f"{exchange_id} 시장 정보 로드 실패: {str(e)}")
    
    def discover_symbol_mappings(self, base_symbols: List[str]):
        """여러 기본 심볼에 대한 CCXT 매핑 자동 탐색"""
        logger.info(f"CCXT 심볼 매핑 탐색 시작: {len(base_symbols)}개 심볼")
        
        # 거래소 시장 정보 로드
        self.load_all_markets()
        
        # 각 심볼별 매핑 탐색
        for base_symbol in base_symbols:
            self.discover_symbol_mapping(base_symbol)
        
        # 매핑 정보 저장
        self.save_symbol_mappings()
        
        logger.info(f"CCXT 심볼 매핑 탐색 완료: {len(self.symbol_mappings)}개 심볼")
    
    def discover_symbol_mapping(self, base_symbol: str):
        """단일 기본 심볼에 대한 CCXT 매핑 탐색"""
        if base_symbol in self.symbol_mappings:
            return
            
        self.symbol_mappings[base_symbol] = {}
        
        # 거래소별 검색 패턴
        search_patterns = {
            "binance": [
                f"{base_symbol}/USDT:USDT",  # 영구 선물
                f"{base_symbol}USDT_PERP",
                f"{base_symbol}/USDT"        # 현물
            ],
            "bybit": [
                f"{base_symbol}USDT",        # 영구 선물
                f"{base_symbol}/USDT"        # 현물
            ],
            "okx": [
                f"{base_symbol}-USDT-SWAP",  # 영구 선물
                f"{base_symbol}/USDT:USDT",
                f"{base_symbol}/USDT"        # 현물
            ],
            "kucoin": [
                f"{base_symbol}USDTM",       # 영구 선물
                f"{base_symbol}-USDT",       # 영구 선물 (다른 형식)
                f"{base_symbol}/USDT"        # 현물
            ],
            "bitget": [
                f"{base_symbol}USDT_UMCBL",  # 영구 선물
                f"{base_symbol}/USDT"        # 현물
            ]
        }
        
        # 거래소 우선순위대로 검색
        for exchange_id in self.exchange_priority:
            if exchange_id not in self.exchanges:
                continue
                
            exchange = self.exchanges[exchange_id]
            patterns = search_patterns.get(exchange_id, [f"{base_symbol}/USDT"])
            
            for pattern in patterns:
                found = False
                for symbol in exchange.symbols:
                    normalized_symbol = symbol.upper()
                    normalized_pattern = pattern.upper()
                    
                    # 정확한 일치 또는 포함 관계 확인
                    if normalized_symbol == normalized_pattern or normalized_pattern in normalized_symbol:
                        self.symbol_mappings[base_symbol][exchange_id] = symbol
                        logger.info(f"심볼 매핑 발견: {base_symbol} -> {exchange_id}:{symbol}")
                        self.supported_symbols.add(base_symbol)
                        found = True
                        break
                
                if found:
                    break
    
    def save_symbol_mappings(self):
        """심볼 매핑 정보 파일로 저장"""
        try:
            filepath = DATA_DIR / "symbol_mappings.json"
            with open(filepath, 'w') as f:
                json.dump({
                    "mappings": self.symbol_mappings,
                    "supported_symbols": list(self.supported_symbols),
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"심볼 매핑 정보 저장 완료: {filepath}")
        except Exception as e:
            logger.error(f"심볼 매핑 정보 저장 실패: {str(e)}")
    
    def load_symbol_mappings(self) -> bool:
        """저장된 심볼 매핑 정보 로드"""
        try:
            filepath = DATA_DIR / "symbol_mappings.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.symbol_mappings = data.get("mappings", {})
                    self.supported_symbols = set(data.get("supported_symbols", []))
                logger.info(f"심볼 매핑 정보 로드 완료: {len(self.symbol_mappings)}개 심볼")
                return True
            return False
        except Exception as e:
            logger.error(f"심볼 매핑 정보 로드 실패: {str(e)}")
            return False
    
    def get_ccxt_symbol(self, base_symbol: str, exchange_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """CCXT 심볼 가져오기 (특정 거래소 또는 우선순위 기반)"""
        # 매핑 정보 확인
        if base_symbol not in self.symbol_mappings:
            self.discover_symbol_mapping(base_symbol)
        
        # 특정 거래소 지정한 경우
        if exchange_id:
            return self.symbol_mappings.get(base_symbol, {}).get(exchange_id), exchange_id
        
        # 거래소 우선순위 기준 탐색
        for ex_id in self.exchange_priority:
            if ex_id in self.symbol_mappings.get(base_symbol, {}):
                return self.symbol_mappings[base_symbol][ex_id], ex_id
        
        return None, None
    
    def fetch_ohlcv(self, base_symbol: str, timeframe: str = '5m', since: Optional[datetime] = None, limit: int = 1000) -> pd.DataFrame:
        """CCXT를 통해 OHLCV 데이터 가져오기"""
        # 심볼 매핑 확인 및 가져오기
        ccxt_symbol, exchange_id = self.get_ccxt_symbol(base_symbol)
        
        if not ccxt_symbol or not exchange_id:
            logger.warning(f"CCXT 심볼 매핑을 찾을 수 없음: {base_symbol}")
            return pd.DataFrame()
        
        # 타임프레임 변환
        ccxt_timeframe = self.timeframe_map.get(timeframe, timeframe)
        
        # since 변환 (timestamp to milliseconds)
        since_ms = int(since.timestamp() * 1000) if since else None
        
        # 데이터 가져오기 시도
        try:
            exchange = self.exchanges[exchange_id]
            
            # OHLCV 데이터 가져오기
            ohlcv = exchange.fetch_ohlcv(ccxt_symbol, ccxt_timeframe, since_ms, limit)
            
            if not ohlcv:
                logger.warning(f"CCXT에서 데이터 없음: {exchange_id}:{ccxt_symbol}")
                return pd.DataFrame()
            
            # DataFrame 변환
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = base_symbol
            df['exchange'] = exchange_id
            
            logger.info(f"CCXT 데이터 수집 성공: {exchange_id}:{ccxt_symbol}, {len(df)}행")
            return df
            
        except Exception as e:
            logger.error(f"CCXT 데이터 수집 실패: {exchange_id}:{ccxt_symbol}, {str(e)}")
            return pd.DataFrame()
    
    def fetch_funding_rate(self, base_symbol: str, timeframe: str = '1h', since: Optional[datetime] = None, limit: int = 500) -> pd.DataFrame:
        """CCXT를 통해 펀딩비 데이터 가져오기 (지원하는 거래소만)"""
        # 펀딩비를 지원하는 거래소들
        funding_supported = ["binance", "bybit", "okx"]
        funding_data = []
        
        # 지원하는 거래소에서 데이터 수집 시도
        for exchange_id in funding_supported:
            if exchange_id not in self.exchanges:
                continue
                
            ccxt_symbol = self.get_ccxt_symbol(base_symbol, exchange_id)[0]
            if not ccxt_symbol:
                continue
                
            try:
                exchange = self.exchanges[exchange_id]
                
                # 1. fetchFundingRateHistory 사용 (최신 데이터)
                if hasattr(exchange, 'fetchFundingRateHistory') and callable(getattr(exchange, 'fetchFundingRateHistory')):
                    since_ms = int(since.timestamp() * 1000) if since else None
                    funding_history = exchange.fetch_funding_rate_history(ccxt_symbol, since_ms, limit)
                    
                    if funding_history:
                        for entry in funding_history:
                            funding_data.append({
                                'datetime': pd.to_datetime(entry['timestamp'], unit='ms'),
                                'symbol': base_symbol,
                                'exchange': exchange_id,
                                'funding_rate': entry['fundingRate']
                            })
                
                # 2. fetchFundingRate 사용 (현재 펀딩비)
                elif hasattr(exchange, 'fetchFundingRate') and callable(getattr(exchange, 'fetchFundingRate')):
                    funding_info = exchange.fetch_funding_rate(ccxt_symbol)
                    if funding_info:
                        funding_data.append({
                            'datetime': pd.to_datetime(funding_info['timestamp'], unit='ms'),
                            'symbol': base_symbol,
                            'exchange': exchange_id,
                            'funding_rate': funding_info['fundingRate']
                        })
            
            except Exception as e:
                logger.error(f"CCXT 펀딩비 데이터 수집 실패: {exchange_id}:{ccxt_symbol}, {str(e)}")
        
        # 수집된 데이터를 DataFrame으로 변환
        if funding_data:
            df = pd.DataFrame(funding_data)
            logger.info(f"CCXT 펀딩비 데이터 수집 성공: {base_symbol}, {len(df)}행")
            return df
        else:
            return pd.DataFrame() 