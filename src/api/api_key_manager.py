"""
API 키 관리 모듈
"""

import time
import random
import threading
from typing import Dict, List, Optional

from ..config import API_KEYS, API_REQUEST_INTERVAL
from ..utils.logger import logger

class APIKeyManager:
    """API 키 관리 클래스"""
    def __init__(self):
        self.tier_keys = API_KEYS
        self.current_key_index = {tier: 0 for tier in API_KEYS.keys()}
        self.key_locks = {tier: threading.Lock() for tier in API_KEYS.keys()}
        self.key_rate_limits = {key: {'last_used': 0, 'count': 0} for tier in API_KEYS.values() for key in tier}
        self.rate_limit_reset_time = time.time() + 60  # 1분 후 리셋
        
    def get_next_key(self, tier: str) -> str:
        """라운드 로빈 방식으로 다음 API 키 반환"""
        if tier not in self.tier_keys or not self.tier_keys[tier]:
            available_tiers = [t for t in self.tier_keys if self.tier_keys[t]]
            if not available_tiers:
                raise ValueError("사용 가능한 API 키가 없습니다")
            tier = random.choice(available_tiers)
            
        with self.key_locks[tier]:
            keys = self.tier_keys[tier]
            current_index = self.current_key_index[tier]
            
            # 현재 시간이 리셋 시간을 초과했는지 확인
            current_time = time.time()
            if current_time > self.rate_limit_reset_time:
                # 사용량 카운터 리셋
                for key_info in self.key_rate_limits.values():
                    key_info['count'] = 0
                self.rate_limit_reset_time = current_time + 60  # 다음 1분 후 리셋
            
            # 사용량이 가장 적은 키를 찾아 반환
            min_count = float('inf')
            selected_key = None
            
            for i in range(len(keys)):
                idx = (current_index + i) % len(keys)
                key = keys[idx]
                
                if self.key_rate_limits[key]['count'] < min_count:
                    min_count = self.key_rate_limits[key]['count']
                    selected_key = key
                    self.current_key_index[tier] = (idx + 1) % len(keys)
            
            # 키 사용량 기록
            self.key_rate_limits[selected_key]['last_used'] = current_time
            self.key_rate_limits[selected_key]['count'] += 1
            
            # API 요청 간격 조절을 위한 대기
            last_used = self.key_rate_limits[selected_key]['last_used']
            if current_time - last_used < API_REQUEST_INTERVAL:
                time.sleep(API_REQUEST_INTERVAL - (current_time - last_used))
            
            return selected_key 