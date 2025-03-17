"""
API 요청 관리 모듈
"""

import time
import random
import requests
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from ..utils.logger import logger
from ..config import API_REQUEST_INTERVAL, ENDPOINTS, API_KEYS
from .api_key_manager import APIKeyManager

class RequestManager:
    def __init__(self):
        """API 요청 관리자 초기화"""
        self.api_key_manager = APIKeyManager()
        self.session = requests.Session()
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_reset = 0
    
    def _wait_for_rate_limit(self):
        """API 요청 간격 조절"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < API_REQUEST_INTERVAL:
            sleep_time = API_REQUEST_INTERVAL - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _handle_rate_limit(self, response: requests.Response):
        """응답의 rate limit 정보 처리"""
        if 'X-RateLimit-Remaining' in response.headers:
            remaining = int(response.headers['X-RateLimit-Remaining'])
            if remaining <= 0:
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                if reset_time > self.rate_limit_reset:
                    self.rate_limit_reset = reset_time
                    logger.warning(f"Rate limit 도달. {reset_time - int(time.time())}초 후 재시도 가능")
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                     headers: Optional[Dict] = None, retries: int = 3) -> Optional[Dict]:
        """API 요청 실행"""
        for attempt in range(retries):
            try:
                self._wait_for_rate_limit()
                
                # API 키 가져오기
                api_key = self.api_key_manager.get_next_key()
                if not api_key:
                    logger.error("사용 가능한 API 키가 없습니다.")
                    return None
                
                # 헤더 설정
                if headers is None:
                    headers = {}
                headers['X-API-KEY'] = api_key
                
                # 요청 실행
                response = self.session.request(
                    method=method,
                    url=endpoint,
                    params=params,
                    headers=headers,
                    timeout=30
                )
                
                # rate limit 처리
                self._handle_rate_limit(response)
                
                # 응답 검증
                response.raise_for_status()
                
                # 응답 데이터 파싱
                data = response.json()
                
                # 에러 체크
                if 'error' in data:
                    logger.error(f"API 에러: {data['error']}")
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)  # 지수 백오프
                        continue
                    return None
                
                return data
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API 요청 실패 (시도 {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # 지수 백오프
                    continue
                return None
        
        return None
    
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000, 
                  since: Optional[int] = None) -> Optional[Dict]:
        """OHLCV 데이터 요청"""
        endpoint = ENDPOINTS['ohlcv']
        params = {
            'symbol': symbol,
            'timeframe': timeframe,
            'limit': limit
        }
        if since:
            params['since'] = since
        
        return self._make_request('GET', endpoint, params=params)
    
    def get_open_interest(self, symbol: str, timeframe: str, limit: int = 1000, 
                         since: Optional[int] = None) -> Optional[Dict]:
        """미체결약정 데이터 요청"""
        endpoint = ENDPOINTS['open_interest']
        params = {
            'symbol': symbol,
            'timeframe': timeframe,
            'limit': limit
        }
        if since:
            params['since'] = since
        
        return self._make_request('GET', endpoint, params=params)
    
    def get_funding_rate(self, symbol: str, timeframe: str, limit: int = 1000, 
                        since: Optional[int] = None) -> Optional[Dict]:
        """펀딩비 데이터 요청"""
        endpoint = ENDPOINTS['funding_rate']
        params = {
            'symbol': symbol,
            'timeframe': timeframe,
            'limit': limit
        }
        if since:
            params['since'] = since
        
        return self._make_request('GET', endpoint, params=params)
    
    def get_predicted_funding_rate(self, symbol: str, timeframe: str, limit: int = 1000, 
                                 since: Optional[int] = None) -> Optional[Dict]:
        """예상 펀딩비 데이터 요청"""
        endpoint = ENDPOINTS['predicted_funding_rate']
        params = {
            'symbol': symbol,
            'timeframe': timeframe,
            'limit': limit
        }
        if since:
            params['since'] = since
        
        return self._make_request('GET', endpoint, params=params)
    
    def get_liquidation(self, symbol: str, timeframe: str, limit: int = 1000, 
                       since: Optional[int] = None) -> Optional[Dict]:
        """청산 데이터 요청"""
        endpoint = ENDPOINTS['liquidation']
        params = {
            'symbol': symbol,
            'timeframe': timeframe,
            'limit': limit
        }
        if since:
            params['since'] = since
        
        return self._make_request('GET', endpoint, params=params)
    
    def get_long_short_ratio(self, symbol: str, timeframe: str, limit: int = 1000, 
                           since: Optional[int] = None) -> Optional[Dict]:
        """롱/숏 비율 데이터 요청"""
        endpoint = ENDPOINTS['long_short_ratio']
        params = {
            'symbol': symbol,
            'timeframe': timeframe,
            'limit': limit
        }
        if since:
            params['since'] = since
        
        return self._make_request('GET', endpoint, params=params)
    
    def get_symbols(self) -> Optional[Dict]:
        """사용 가능한 심볼 목록 요청"""
        endpoint = ENDPOINTS['symbols']
        return self._make_request('GET', endpoint)
    
    def get_timeframes(self) -> Optional[Dict]:
        """사용 가능한 시간대 목록 요청"""
        endpoint = ENDPOINTS['timeframes']
        return self._make_request('GET', endpoint) 