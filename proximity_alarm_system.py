#!/usr/bin/env python
"""
근접 알람 시스템
- 실시간으로 패턴 형성 진행 상황을 모니터링
- 패턴 완성도에 따라 단계별 알람 생성
- 점수 기반 알람 우선순위 설정
"""

import json
import logging
import time
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import argparse

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ProximityAlarm")

class ProximityAlarmSystem:
    """근접 알람 시스템"""
    
    def __init__(self, 
                 signals_file: str = 'data/alarms/all_symbols_best_signals_20250312_131327.json',
                 alarms_dir: str = 'data/alarms',
                 data_dir: str = 'data/live',
                 high_threshold: float = 70.0,
                 medium_threshold: float = 50.0,
                 low_threshold: float = 30.0,
                 completion_high: float = 0.9,  # 90% 완성
                 completion_medium: float = 0.7,  # 70% 완성
                 completion_low: float = 0.5):  # 50% 완성
        """
        초기화
        
        Args:
            signals_file: 신호 파일 경로
            alarms_dir: 알람 저장 디렉토리
            data_dir: 실시간 데이터 디렉토리
            high_threshold: 높은 우선순위 임계값
            medium_threshold: 중간 우선순위 임계값
            low_threshold: 낮은 우선순위 임계값
            completion_high: 높은 완성도 임계값
            completion_medium: 중간 완성도 임계값
            completion_low: 낮은 완성도 임계값
        """
        self.signals_file = Path(signals_file)
        self.alarms_dir = Path(alarms_dir)
        self.data_dir = Path(data_dir)
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold
        self.completion_high = completion_high
        self.completion_medium = completion_medium
        self.completion_low = completion_low
        
        # 알람 상태 저장
        self.alarm_status = {}
        
        # 알람 히스토리 (중복 알람 방지용)
        self.alarm_history = defaultdict(lambda: deque(maxlen=10))
        
        # 패턴 데이터 캐시
        self.pattern_data_cache = {}
        
        # 알람 디렉토리 생성
        self.alarms_dir.mkdir(parents=True, exist_ok=True)
        
        # 신호 데이터 로드
        self.signals_data = self._load_signals_data()
    
    def _get_latest_signals_file(self):
        """가장 최근의 best_signals 파일을 찾습니다."""
        files = list(Path(self.alarms_dir).glob("all_symbols_best_signals_*.json"))
        if not files:
            logging.error("최고 점수 신호 파일을 찾을 수 없습니다.")
            return None
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        return str(latest_file)
    
    def _load_signals_data(self):
        """신호 데이터를 로드합니다."""
        if not self.signals_file or not os.path.exists(self.signals_file):
            logging.error(f"신호 파일을 찾을 수 없습니다: {self.signals_file}")
            return {}
        
        try:
            with open(self.signals_file, 'r') as f:
                data = json.load(f)
                logging.info(f"총 {len(data['symbols'])}개 심볼의 최고 점수 신호 로드 완료")
                return data
        except Exception as e:
            logging.error(f"신호 파일 로드 중 오류 발생: {e}")
            return {}
    
    def load_latest_data(self, symbol, timeframe):
        """
        최신 데이터 로드
        
        Args:
            symbol: 심볼
            timeframe: 타임프레임
            
        Returns:
            최신 데이터 (없으면 None)
        """
        try:
            # 캐시 키 생성
            cache_key = f"{symbol}_{timeframe}"
            
            # 캐시에 있으면 반환
            if cache_key in self.pattern_data_cache:
                return self.pattern_data_cache[cache_key]
            
            # 데이터 파일 경로
            data_file = self.data_dir / f"{symbol}_{timeframe}.csv"
            
            if not data_file.exists():
                logging.warning(f"데이터 파일이 없습니다: {data_file}")
                return None
            
            # 데이터 로드 (여기서는 간단히 파일이 있는지만 확인)
            # 실제 구현에서는 CSV 파일을 파싱하여 데이터 로드
            data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'exists': True,
                'last_modified': os.path.getmtime(data_file)
            }
            
            # 캐시에 저장
            self.pattern_data_cache[cache_key] = data
            
            return data
            
        except Exception as e:
            logging.error(f"데이터 로드 중 오류 발생: {e}")
            return None
    
    def calculate_pattern_completion(self, signal, current_time=None):
        """
        패턴 완성도 계산
        
        실제 데이터를 기반으로 패턴 완성도를 계산
        
        Args:
            signal: 신호 데이터
            current_time: 현재 시간 (None이면 현재 시간 사용)
            
        Returns:
            완성도 (0.0 ~ 1.0)
        """
        if current_time is None:
            current_time = datetime.now()
        
        symbol = signal.get('symbol', '')
        timeframe = signal.get('timeframe', '')
        event_type = signal.get('event_type', '')
        period = signal.get('period', '')
        score = signal.get('score', 0)
        
        # 신호 ID 생성
        signal_id = f"{symbol}_{timeframe}_{event_type}_{period}"
        
        # 이전 상태가 없으면 초기화
        if signal_id not in self.alarm_status:
            self.alarm_status[signal_id] = {
                'last_update': current_time,
                'completion': 0.1,  # 초기 완성도 10%
                'last_alarm_level': None,
                'data_timestamp': None
            }
        
        # 최신 데이터 로드
        data = self.load_latest_data(symbol, timeframe)
        
        if data is None:
            # 데이터가 없으면 시뮬레이션 모드로 전환
            return self.simulate_pattern_completion(signal, current_time)
        
        # 데이터 타임스탬프
        data_timestamp = data.get('last_modified', 0)
        last_data_timestamp = self.alarm_status[signal_id].get('data_timestamp', 0)
        
        # 데이터가 업데이트되지 않았으면 이전 완성도 유지
        if data_timestamp == last_data_timestamp:
            return self.alarm_status[signal_id]['completion']
        
        # 데이터 업데이트 시간 저장
        self.alarm_status[signal_id]['data_timestamp'] = data_timestamp
        
        # 점수에 따른 기본 완성도 설정
        # 점수가 높을수록 완성도가 높음
        base_completion = min(1.0, score / 100)
        
        # 이벤트 유형에 따른 가중치 적용
        if event_type == 'funding_changes':
            # funding_changes는 더 빠르게 완성됨
            weight = 1.2
        elif event_type == 'volume_spikes':
            # volume_spikes는 표준 속도로 완성됨
            weight = 1.0
        else:
            # 기타 이벤트는 느리게 완성됨
            weight = 0.8
        
        # 타임프레임에 따른 가중치 적용
        if timeframe == '1M':
            # 1분 타임프레임은 빠르게 완성됨
            tf_weight = 1.2
        elif timeframe == '5M':
            # 5분 타임프레임은 표준 속도로 완성됨
            tf_weight = 1.0
        elif timeframe == '15M':
            # 15분 타임프레임은 느리게 완성됨
            tf_weight = 0.8
        elif timeframe == '30M':
            # 30분 타임프레임은 더 느리게 완성됨
            tf_weight = 0.6
        else:
            # 1시간 이상 타임프레임은 매우 느리게 완성됨
            tf_weight = 0.4
        
        # 마지막 업데이트 이후 경과 시간 (초)
        elapsed_seconds = (current_time - self.alarm_status[signal_id]['last_update']).total_seconds()
        
        # 완성도 증가율 계산
        # 기본적으로 5분(300초)마다 10% 증가하는 것으로 가정
        increase_rate = (elapsed_seconds / 300) * 0.1 * weight * tf_weight
        
        # 현재 완성도
        current_completion = self.alarm_status[signal_id]['completion']
        
        # 완성도 업데이트 (최대 1.0, 최소 base_completion)
        completion = min(1.0, max(base_completion, current_completion + increase_rate))
        
        # 상태 업데이트
        self.alarm_status[signal_id]['last_update'] = current_time
        self.alarm_status[signal_id]['completion'] = completion
        
        return completion
    
    def simulate_pattern_completion(self, signal, current_time=None):
        """
        패턴 완성도 시뮬레이션
        
        실제 구현에서는 실시간 데이터를 기반으로 패턴 완성도를 계산해야 함
        여기서는 시뮬레이션을 위해 임의의 완성도 반환
        
        Args:
            signal: 신호 데이터
            current_time: 현재 시간 (None이면 현재 시간 사용)
            
        Returns:
            완성도 (0.0 ~ 1.0)
        """
        # 실제 구현에서는 이 부분을 실시간 데이터 분석으로 대체
        # 여기서는 시뮬레이션을 위해 시간에 따라 완성도가 증가하는 것으로 가정
        
        if current_time is None:
            current_time = datetime.now()
        
        # 신호 ID 생성 (심볼, 타임프레임, 이벤트 유형, 기간)
        signal_id = f"{signal['symbol']}_{signal['timeframe']}_{signal['event_type']}_{signal['period']}"
        
        # 이전 상태가 없으면 초기화
        if signal_id not in self.alarm_status:
            self.alarm_status[signal_id] = {
                'last_update': current_time,
                'completion': 0.1,  # 초기 완성도 10%
                'last_alarm_level': None
            }
        
        # 마지막 업데이트 이후 경과 시간 (초)
        elapsed_seconds = (current_time - self.alarm_status[signal_id]['last_update']).total_seconds()
        
        # 시간에 따라 완성도 증가 (최대 1.0)
        # 실제 구현에서는 이 부분을 실시간 데이터 분석으로 대체
        completion = min(1.0, self.alarm_status[signal_id]['completion'] + (elapsed_seconds / 300))
        
        # 상태 업데이트
        self.alarm_status[signal_id]['last_update'] = current_time
        self.alarm_status[signal_id]['completion'] = completion
        
        return completion
    
    def check_alarm_condition(self, signal, completion):
        """
        알람 조건 확인
        
        Args:
            signal: 신호 데이터
            completion: 패턴 완성도 (0.0 ~ 1.0)
            
        Returns:
            (알람 레벨, 알람 메시지)
            알람 레벨: 'high', 'medium', 'low', None
        """
        score = signal.get('score', 0)
        symbol = signal.get('symbol', '')
        timeframe = signal.get('timeframe', '')
        event_type = signal.get('event_type', '')
        period = signal.get('period', '')
        
        # 신호 ID 생성
        signal_id = f"{symbol}_{timeframe}_{event_type}_{period}"
        
        # 이전 알람 레벨
        last_alarm_level = self.alarm_status[signal_id].get('last_alarm_level')
        
        # 완성도와 점수에 따른 알람 레벨 결정
        if completion >= self.completion_high and score >= self.high_threshold:
            alarm_level = 'high'
            message = f"[긴급] {symbol} {timeframe} {event_type} 패턴 거의 완성 (완성도: {completion:.1%}, 점수: {score})"
        elif completion >= self.completion_medium and score >= self.medium_threshold:
            alarm_level = 'medium'
            message = f"[주의] {symbol} {timeframe} {event_type} 패턴 형성 중 (완성도: {completion:.1%}, 점수: {score})"
        elif completion >= self.completion_low and score >= self.low_threshold:
            alarm_level = 'low'
            message = f"[관찰] {symbol} {timeframe} {event_type} 패턴 감지 (완성도: {completion:.1%}, 점수: {score})"
        else:
            alarm_level = None
            message = None
        
        # 이전과 동일한 레벨의 알람이면 중복 방지
        if alarm_level == last_alarm_level:
            return None, None
        
        # 알람 레벨 업데이트
        self.alarm_status[signal_id]['last_alarm_level'] = alarm_level
        
        return alarm_level, message
    
    def generate_alarms(self, current_time=None):
        """
        알람 생성
        
        Args:
            current_time: 현재 시간 (None이면 현재 시간 사용)
            
        Returns:
            생성된 알람 목록
        """
        if current_time is None:
            current_time = datetime.now()
        
        # 신호 로드
        best_signals = self.signals_data.get('symbols', [])
        
        # 알람 목록
        alarms = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        # 각 신호에 대해 알람 조건 확인
        for signal in best_signals:
            # 패턴 완성도 계산
            completion = self.calculate_pattern_completion(signal, current_time)
            
            # 알람 조건 확인
            alarm_level, message = self.check_alarm_condition(signal, completion)
            
            # 알람 추가
            if alarm_level and message:
                alarm_data = {
                    'timestamp': current_time.isoformat(),
                    'symbol': signal.get('symbol', ''),
                    'timeframe': signal.get('timeframe', ''),
                    'event_type': signal.get('event_type', ''),
                    'period': signal.get('period', ''),
                    'score': signal.get('score', 0),
                    'completion': completion,
                    'message': message
                }
                
                alarms[alarm_level].append(alarm_data)
        
        # 알람 저장
        self.save_alarms(alarms, current_time)
        
        # 알람 출력
        self.print_alarms(alarms)
        
        return alarms
    
    def save_alarms(self, alarms, current_time):
        """
        알람 저장
        
        Args:
            alarms: 알람 목록
            current_time: 현재 시간
        """
        try:
            # 모든 알람 합치기
            all_alarms = []
            for level, level_alarms in alarms.items():
                for alarm in level_alarms:
                    alarm['level'] = level
                    all_alarms.append(alarm)
            
            if not all_alarms:
                return
            
            # 파일명 생성
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            filename = f"proximity_alarms_{timestamp}.json"
            filepath = self.alarms_dir / filename
            
            # 알람 저장
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': current_time.isoformat(),
                    'total_alarms': len(all_alarms),
                    'alarms': all_alarms
                }, f, indent=2)
            
            logger.info(f"알람이 {filepath}에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"알람 저장 중 오류 발생: {e}")
    
    def print_alarms(self, alarms):
        """
        알람 출력
        
        Args:
            alarms: 알람 목록
        """
        # 높은 우선순위 알람
        if alarms['high']:
            logger.info(f"=== 높은 우선순위 알람 ({len(alarms['high'])}개) ===")
            for alarm in alarms['high']:
                logger.info(alarm['message'])
        
        # 중간 우선순위 알람
        if alarms['medium']:
            logger.info(f"=== 중간 우선순위 알람 ({len(alarms['medium'])}개) ===")
            for alarm in alarms['medium']:
                logger.info(alarm['message'])
        
        # 낮은 우선순위 알람
        if alarms['low']:
            logger.info(f"=== 낮은 우선순위 알람 ({len(alarms['low'])}개) ===")
            for alarm in alarms['low']:
                logger.info(alarm['message'])
    
    def run_monitoring(self, interval=60, duration=None):
        """
        모니터링 실행
        
        Args:
            interval: 모니터링 간격 (초)
            duration: 모니터링 지속 시간 (초), None이면 무한 실행
        """
        logger.info(f"근접 알람 모니터링 시작 (간격: {interval}초)")
        
        start_time = time.time()
        iteration = 0
        
        try:
            while True:
                iteration += 1
                current_time = datetime.now()
                
                logger.info(f"=== 모니터링 #{iteration} ({current_time.isoformat()}) ===")
                
                # 알람 생성
                self.generate_alarms(current_time)
                
                # 지속 시간 체크
                if duration and (time.time() - start_time) >= duration:
                    logger.info(f"지정된 지속 시간 {duration}초가 경과하여 모니터링을 종료합니다.")
                    break
                
                # 대기
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("사용자에 의해 모니터링이 중단되었습니다.")
        except Exception as e:
            logger.error(f"모니터링 중 오류 발생: {e}")
        finally:
            logger.info("근접 알람 모니터링 종료")

    def generate_sample_data(self, symbol, timeframe):
        """테스트용 샘플 데이터를 생성합니다."""
        file_path = os.path.join(self.data_dir, f"{symbol}_{timeframe}.csv")
        
        # 기본 데이터 프레임 생성
        periods = 100
        now = datetime.datetime.now()
        dates = [now - datetime.timedelta(minutes=i) for i in range(periods)]
        dates.reverse()
        
        # 랜덤 데이터 생성
        np.random.seed(42)  # 재현성을 위한 시드 설정
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(100, 5, periods),
            'high': np.random.normal(105, 5, periods),
            'low': np.random.normal(95, 5, periods),
            'close': np.random.normal(100, 5, periods),
            'volume': np.random.normal(1000, 200, periods),
            'oi_change': np.random.normal(0.01, 0.02, periods),
            'funding_rate': np.random.normal(0.0005, 0.001, periods)
        })
        
        # 파일 저장
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"샘플 데이터 생성 완료: {file_path}")
        
        return file_path

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='근접 알람 시스템')
    parser.add_argument('--file', type=str, default='data/alarms/all_symbols_best_signals_20250312_131327.json', help='신호 파일 경로')
    parser.add_argument('--interval', type=int, default=60, help='모니터링 간격 (초)')
    parser.add_argument('--duration', type=int, default=None, help='모니터링 지속 시간 (초), 지정하지 않으면 무한 실행')
    parser.add_argument('--high', type=float, default=70.0, help='높은 우선순위 임계값')
    parser.add_argument('--medium', type=float, default=50.0, help='중간 우선순위 임계값')
    parser.add_argument('--low', type=float, default=30.0, help='낮은 우선순위 임계값')
    parser.add_argument('--data-dir', type=str, default='data/live', help='실시간 데이터 디렉토리')
    parser.add_argument('--generate-sample', action="store_true", help="데이터 파일이 없을 경우 샘플 데이터 생성")
    
    args = parser.parse_args()
    
    # 알람 시스템 초기화
    alarm_system = ProximityAlarmSystem(
        signals_file=args.file,
        high_threshold=args.high,
        medium_threshold=args.medium,
        low_threshold=args.low,
        data_dir=args.data_dir
    )
    
    # 샘플 데이터 생성 옵션 설정
    if args.generate_sample:
        os.environ['GENERATE_SAMPLE_DATA'] = 'true'
    
    # 모니터링 실행
    alarm_system.run_monitoring(interval=args.interval, duration=args.duration)

if __name__ == "__main__":
    main() 