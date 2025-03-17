"""
암호화폐 데이터 통합 관리 시스템 메인 모듈
"""

import os
import sys
import time
import signal
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .utils.logger import logger
from .config import COLLECTION_INTERVAL, CCXT_EXCHANGES
from .api.request_manager import RequestManager
from .api.ccxt_manager import CcxtManager
from .data.processor import (
    get_simple_symbol, exponential_backoff, extract_data_from_response,
    normalize_columns, validate_data, merge_dataframes
)
from .storage.data_manager import DataManager

class CryptoDataManager:
    def __init__(self):
        """암호화폐 데이터 관리자 초기화"""
        self.request_manager = RequestManager()
        self.ccxt_manager = CcxtManager()
        self.data_manager = DataManager()
        self.running = False
        self.threads: List[threading.Thread] = []
        self.symbols: List[str] = []
        self.timeframes: List[str] = []
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """시그널 핸들러"""
        logger.info("종료 시그널 수신. 안전하게 종료합니다...")
        self.running = False
    
    def initialize(self):
        """시스템 초기화"""
        try:
            # CCXT 거래소 초기화
            self.ccxt_manager.init_exchanges()
            
            # 사용 가능한 심볼 목록 로드
            symbols_response = self.request_manager.get_symbols()
            if symbols_response and 'data' in symbols_response:
                self.symbols = [get_simple_symbol(s) for s in symbols_response['data']]
                logger.info(f"로드된 심볼 수: {len(self.symbols)}")
            
            # 사용 가능한 시간대 목록 로드
            timeframes_response = self.request_manager.get_timeframes()
            if timeframes_response and 'data' in timeframes_response:
                self.timeframes = timeframes_response['data']
                logger.info(f"로드된 시간대 수: {len(self.timeframes)}")
            
            return True
            
        except Exception as e:
            logger.error(f"초기화 중 오류 발생: {str(e)}")
            return False
    
    def collect_historical_data(self, symbol: str, timeframe: str, 
                              start_time: Optional[int] = None, 
                              end_time: Optional[int] = None):
        """과거 데이터 수집"""
        try:
            # CCXT를 통한 과거 데이터 수집
            ccxt_data = self.ccxt_manager.fetch_ohlcv(symbol, timeframe, start_time, end_time)
            if not ccxt_data.empty:
                self.data_manager.save_long_term_data(ccxt_data, symbol, 'ohlcv')
            
            # API를 통한 과거 데이터 수집
            data_types = ['ohlcv', 'open_interest', 'funding_rate', 
                         'predicted_funding_rate', 'liquidation', 'long_short_ratio']
            
            for data_type in data_types:
                attempt = 0
                while True:
                    try:
                        # API 요청
                        response = getattr(self.request_manager, f"get_{data_type}")(
                            symbol, timeframe, since=start_time
                        )
                        
                        if not response:
                            break
                        
                        # 데이터 추출 및 처리
                        history = extract_data_from_response(response, data_type)
                        if not history:
                            break
                        
                        # DataFrame 생성 및 정규화
                        df = pd.DataFrame(history)
                        df = normalize_columns(df, data_type)
                        df = validate_data(df, data_type)
                        
                        # 데이터 저장
                        self.data_manager.save_long_term_data(df, symbol, data_type)
                        
                        # 다음 페이지 확인
                        if len(history) < 1000:  # 마지막 페이지
                            break
                        
                        # 다음 요청을 위한 시간 설정
                        start_time = history[-1]['t'] + 1
                        
                    except Exception as e:
                        logger.error(f"{symbol} {data_type} 데이터 수집 중 오류: {str(e)}")
                        attempt += 1
                        if attempt >= 3:  # 최대 3번 재시도
                            break
                        time.sleep(exponential_backoff(attempt))
            
            logger.info(f"{symbol} 과거 데이터 수집 완료")
            
        except Exception as e:
            logger.error(f"{symbol} 과거 데이터 수집 중 오류: {str(e)}")
    
    def collect_realtime_data(self, symbol: str):
        """실시간 데이터 수집"""
        while self.running:
            try:
                all_data_frames = {}
                
                # 각 데이터 타입별 데이터 수집
                data_types = ['ohlcv', 'open_interest', 'funding_rate', 
                             'predicted_funding_rate', 'liquidation', 'long_short_ratio']
                
                for data_type in data_types:
                    response = getattr(self.request_manager, f"get_{data_type}")(
                        symbol, '1m', limit=1
                    )
                    
                    if response:
                        history = extract_data_from_response(response, data_type)
                        if history:
                            df = pd.DataFrame(history)
                            df = normalize_columns(df, data_type)
                            df = validate_data(df, data_type)
                            all_data_frames[data_type] = df
                
                # 데이터 병합 및 저장
                if all_data_frames:
                    merged_df = merge_dataframes(all_data_frames, symbol)
                    if not merged_df.empty:
                        self.data_manager.save_data(merged_df, symbol, 'realtime')
                
                # 다음 수집까지 대기
                time.sleep(COLLECTION_INTERVAL)
                
            except Exception as e:
                logger.error(f"{symbol} 실시간 데이터 수집 중 오류: {str(e)}")
                time.sleep(5)  # 오류 발생 시 5초 대기
    
    def start(self):
        """시스템 시작"""
        if not self.initialize():
            logger.error("시스템 초기화 실패")
            return
        
        self.running = True
        
        # 과거 데이터 수집 스레드 시작
        for symbol in self.symbols:
            thread = threading.Thread(
                target=self.collect_historical_data,
                args=(symbol, '1h'),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        # 실시간 데이터 수집 스레드 시작
        for symbol in self.symbols:
            thread = threading.Thread(
                target=self.collect_realtime_data,
                args=(symbol,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        # 메인 스레드에서 종료 대기
        while self.running:
            time.sleep(1)
        
        # 스레드 종료 대기
        for thread in self.threads:
            thread.join()
        
        logger.info("시스템이 안전하게 종료되었습니다.")

def main():
    """메인 함수"""
    try:
        manager = CryptoDataManager()
        manager.start()
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 