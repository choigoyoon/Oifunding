import pandas as pd
import numpy as np
import json
import os
import glob
import time
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_alarm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoAlarmSystem")

class CryptoAlarmSystem:
    """암호화폐 알람 시스템 - 승률 기반 신호 생성"""
    
    def __init__(self, 
                 data_dir: str = 'data/live', 
                 results_dir: str = 'data/results',
                 output_dir: str = 'data/alarms',
                 min_score: float = 70.0,
                 lookforward_periods: List[int] = [5, 15, 30, 60]):
        """
        초기화
        
        Args:
            data_dir: CSV 데이터 디렉토리
            results_dir: JSON 결과 디렉토리
            output_dir: 알람 출력 디렉토리
            min_score: 최소 가능성 점수 (기본값: 70.0)
            lookforward_periods: 미래 가격 변화 확인 기간 (분 단위)
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.min_score = min_score
        self.lookforward_periods = lookforward_periods
        
        # 결과 저장 변수
        self.win_rates = {}
        self.probability_scores = {}
        self.all_signals = []
        self.perfect_signals = []  # 100점 만점 신호
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self, symbol: str) -> Tuple[pd.DataFrame, Dict]:
        """CSV 데이터와 JSON 결과 로드"""
        # CSV 데이터 로드
        csv_path = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} 파일을 찾을 수 없습니다.")
        
        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # JSON 결과 로드 - '_changes' 접미사 추가
        json_path = os.path.join(self.results_dir, f"{symbol}_changes.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} 파일을 찾을 수 없습니다.")
        
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        return df, json_data
    
    def calculate_win_rates(self, symbol: str) -> Dict:
        """알람 승률 계산"""
        try:
            df, json_data = self.load_data(symbol)
            
            # 결과 저장 딕셔너리
            win_rates = {
                'oi_changes': {},
                'funding_changes': {},
                'volume_spikes': {}
            }
            
            # 각 타임프레임별 분석
            for tf in json_data['results'].keys():
                win_rates[tf] = {}
                
                # 1. OI 변화 승률 계산
                oi_changes = json_data['results'][tf].get('oi_changes', [])
                if oi_changes:
                    win_rates[tf]['oi_changes'] = self._calculate_event_win_rate(
                        df, oi_changes, 'oi', self.lookforward_periods
                    )
                
                # 2. 자금 비율 변화 승률 계산
                funding_changes = json_data['results'][tf].get('funding_changes', [])
                if funding_changes:
                    win_rates[tf]['funding_changes'] = self._calculate_event_win_rate(
                        df, funding_changes, 'funding', self.lookforward_periods
                    )
                
                # 3. 거래량 스파이크 승률 계산
                volume_spikes = json_data['results'][tf].get('volume_spikes', [])
                if volume_spikes:
                    win_rates[tf]['volume_spikes'] = self._calculate_event_win_rate(
                        df, volume_spikes, 'volume', self.lookforward_periods
                    )
            
            self.win_rates[symbol] = win_rates
            return win_rates
        except Exception as e:
            logger.error(f"{symbol} 승률 계산 중 오류 발생: {e}")
            return {}
    
    def _calculate_event_win_rate(self, df: pd.DataFrame, events: List[Dict], 
                                 event_type: str, lookforward_periods: List[int]) -> Dict:
        """특정 이벤트 유형의 승률 계산"""
        results = {period: {'up': {'wins': 0, 'total': 0, 'win_rate': 0},
                           'down': {'wins': 0, 'total': 0, 'win_rate': 0}}
                  for period in lookforward_periods}
        
        for event in events:
            # 이벤트 시간 및 방향 추출
            try:
                # 타임스탬프가 인덱스 형태인 경우
                if event['timestamp'].isdigit():
                    event_idx = int(event['timestamp'])
                    if event_idx >= len(df):
                        continue
                    event_time = df.iloc[event_idx]['datetime']
                else:
                    # 타임스탬프가 ISO 형식인 경우
                    event_time = pd.to_datetime(event['timestamp'])
                
                # 가장 가까운 데이터 포인트 찾기
                closest_idx = df['datetime'].searchsorted(event_time)
                if closest_idx >= len(df):
                    closest_idx = len(df) - 1
                
                event_price = event['price']
                event_direction = event.get('direction', '')
                
                # 각 기간별 승률 계산
                for period in lookforward_periods:
                    if closest_idx + period >= len(df):
                        continue
                    
                    future_price = df.iloc[closest_idx + period]['close']
                    price_change_pct = (future_price - event_price) / event_price * 100
                    
                    # 방향에 따른 승패 판정
                    if event_direction == 'up':
                        results[period]['up']['total'] += 1
                        if price_change_pct > 0:  # 상승 예측이 맞았을 경우
                            results[period]['up']['wins'] += 1
                    elif event_direction == 'down':
                        results[period]['down']['total'] += 1
                        if price_change_pct < 0:  # 하락 예측이 맞았을 경우
                            results[period]['down']['wins'] += 1
                    elif event_direction == 'positive' and event_type == 'funding':
                        results[period]['up']['total'] += 1
                        if price_change_pct > 0:  # 긍정적 자금 비율 -> 상승
                            results[period]['up']['wins'] += 1
                    elif event_direction == 'negative' and event_type == 'funding':
                        results[period]['down']['total'] += 1
                        if price_change_pct < 0:  # 부정적 자금 비율 -> 하락
                            results[period]['down']['wins'] += 1
            except Exception as e:
                logger.warning(f"이벤트 처리 중 오류 발생: {e}")
                continue
        
        # 승률 계산
        for period in lookforward_periods:
            for direction in ['up', 'down']:
                total = results[period][direction]['total']
                if total > 0:
                    win_rate = results[period][direction]['wins'] / total * 100
                    results[period][direction]['win_rate'] = round(win_rate, 2)
        
        return results
    
    def calculate_probability_score(self, symbol: str) -> Dict:
        """알람 가능성 점수 계산 (100점 만점)"""
        if symbol not in self.win_rates:
            self.calculate_win_rates(symbol)
        
        if symbol not in self.win_rates or not self.win_rates[symbol]:
            return {}
        
        win_rates = self.win_rates[symbol]
        
        # 가능성 점수 계산 (각 지표별 가중치 적용)
        probability_scores = {}
        
        for tf in win_rates.keys():
            probability_scores[tf] = {}
            
            # 각 이벤트 유형별 점수 계산
            for event_type in ['oi_changes', 'funding_changes', 'volume_spikes']:
                if event_type in win_rates[tf]:
                    event_scores = {}
                    
                    # 각 기간별 점수 계산
                    for period, directions in win_rates[tf][event_type].items():
                        period_score = 0
                        total_samples = 0
                        
                        # 상승/하락 방향별 점수 계산
                        for direction, stats in directions.items():
                            if stats['total'] > 0:
                                # 승률 기반 점수 (최대 70점)
                                win_rate_score = min(70, stats['win_rate'] * 0.7)
                                
                                # 샘플 수 기반 신뢰도 점수 (최대 30점)
                                confidence_score = min(30, stats['total'] * 3)
                                
                                # 방향별 총점
                                direction_score = win_rate_score + confidence_score
                                
                                # 가중 평균을 위한 누적
                                period_score += direction_score * stats['total']
                                total_samples += stats['total']
                        
                        # 가중 평균 계산
                        if total_samples > 0:
                            event_scores[period] = round(period_score / total_samples, 2)
                        else:
                            event_scores[period] = 0
                    
                    probability_scores[tf][event_type] = event_scores
        
        self.probability_scores[symbol] = probability_scores
        return probability_scores
    
    def get_best_signals(self, symbol: str) -> List[Dict]:
        """가장 높은 가능성 점수를 가진 신호 추출"""
        if symbol not in self.probability_scores:
            self.calculate_probability_score(symbol)
        
        if symbol not in self.probability_scores or not self.probability_scores[symbol]:
            return []
        
        best_signals = []
        
        for tf, events in self.probability_scores[symbol].items():
            for event_type, periods in events.items():
                for period, score in periods.items():
                    if score >= self.min_score:
                        best_signals.append({
                            'symbol': symbol,
                            'timeframe': tf,
                            'event_type': event_type,
                            'period': period,
                            'score': score
                        })
        
        # 점수 기준 내림차순 정렬
        best_signals.sort(key=lambda x: x['score'], reverse=True)
        return best_signals
    
    def analyze_all_symbols(self) -> Dict:
        """모든 심볼 분석 및 결과 반환"""
        # 모든 JSON 파일 찾기
        json_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        # '_changes' 접미사 제거
        symbols = [os.path.basename(f).split('_changes.')[0] for f in json_files]
        
        all_results = {}
        self.all_signals = []
        
        for symbol in symbols:
            try:
                win_rates = self.calculate_win_rates(symbol)
                if not win_rates:
                    continue
                    
                probability_scores = self.calculate_probability_score(symbol)
                if not probability_scores:
                    continue
                    
                best_signals = self.get_best_signals(symbol)
                
                all_results[symbol] = {
                    'win_rates': win_rates,
                    'probability_scores': probability_scores,
                    'best_signals': best_signals
                }
                
                # 전체 신호 리스트에 추가
                for signal in best_signals:
                    self.all_signals.append(signal)
                
                logger.info(f"{symbol} 분석 완료: 최고 점수 신호 {len(best_signals)}개 발견")
            except Exception as e:
                logger.error(f"{symbol} 분석 중 오류 발생: {e}")
        
        # 점수 기준 내림차순 정렬
        self.all_signals.sort(key=lambda x: x['score'], reverse=True)
        
        # 100점 만점 신호 추출
        self.perfect_signals = [s for s in self.all_signals if s['score'] == 100.0]
        
        return all_results
    
    def save_results(self, output_file: str = None) -> None:
        """분석 결과 저장"""
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'win_rate_analysis.json')
            
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {}
        }
        
        for symbol in self.win_rates.keys():
            if symbol not in self.probability_scores:
                continue
                
            all_results['symbols'][symbol] = {
                'win_rates': self.win_rates.get(symbol, {}),
                'probability_scores': self.probability_scores.get(symbol, {}),
                'best_signals': self.get_best_signals(symbol)
            }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"분석 결과가 {output_file}에 저장되었습니다.")
    
    def generate_alarms(self, top_n: int = 20) -> List[Dict]:
        """상위 N개 알람 생성"""
        if not self.all_signals:
            self.analyze_all_symbols()
            
        top_signals = self.all_signals[:top_n]
        
        # 알람 파일 생성
        alarm_file = os.path.join(self.output_dir, f'alarms_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(alarm_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'alarms': top_signals
            }, f, indent=2)
            
        logger.info(f"상위 {len(top_signals)}개 알람이 {alarm_file}에 저장되었습니다.")
        return top_signals
    
    def print_symbol_analysis(self, symbol: str) -> None:
        """특정 심볼의 분석 결과 출력"""
        if symbol not in self.win_rates:
            self.calculate_win_rates(symbol)
            self.calculate_probability_score(symbol)
            
        if symbol not in self.win_rates or not self.win_rates[symbol]:
            logger.warning(f"{symbol} 분석 결과가 없습니다.")
            return
            
        win_rates = self.win_rates[symbol]
        probability_scores = self.probability_scores[symbol]
        best_signals = self.get_best_signals(symbol)
        
        print("=" * 50)
        print(f"{symbol} 승률 정보")
        print("=" * 50)
        
        # 타임프레임별 승률 정보
        for tf, events in win_rates.items():
            if tf in ['oi_changes', 'funding_changes', 'volume_spikes']:
                continue
            
            print(f"\n타임프레임: {tf}")
            
            # 이벤트 유형별 승률
            for event_type, periods in events.items():
                print(f"\n  {event_type}:")
                
                for period, directions in periods.items():
                    print(f"    {period}분 후:")
                    
                    for direction, stats in directions.items():
                        if stats['total'] > 0:
                            print(f"      {direction}: {stats['wins']}/{stats['total']} ({stats['win_rate']}%)")
        
        # 가능성 점수 출력
        print("\n" + "=" * 50)
        print(f"{symbol} 가능성 점수")
        print("=" * 50)
        
        for tf, events in probability_scores.items():
            print(f"\n타임프레임: {tf}")
            
            for event_type, periods in events.items():
                print(f"\n  {event_type}:")
                
                for period, score in periods.items():
                    print(f"    {period}분 후: {score}점")
        
        # 최고 점수 신호 출력
        print("\n" + "=" * 50)
        print(f"{symbol} 최고 점수 신호 ({self.min_score}점 이상)")
        print("=" * 50)
        
        for i, signal in enumerate(best_signals, 1):
            print(f"\n{i}. 타임프레임: {signal['timeframe']}")
            print(f"   이벤트 유형: {signal['event_type']}")
            print(f"   기간: {signal['period']}분")
            print(f"   점수: {signal['score']}점")
    
    def print_top_signals_analysis(self, top_n: int = 100) -> None:
        """상위 신호 분석 결과 출력"""
        if not self.all_signals:
            self.analyze_all_symbols()
            
        signals_to_analyze = self.all_signals[:top_n]
        
        # 상위 10개 신호 출력
        print("=" * 50)
        print(f"상위 10개 신호 (전체 심볼)")
        print("=" * 50)
        
        for i, signal in enumerate(signals_to_analyze[:10], 1):
            print(f"\n{i}. 심볼: {signal['symbol']}")
            print(f"   타임프레임: {signal['timeframe']}")
            print(f"   이벤트 유형: {signal['event_type']}")
            print(f"   기간: {signal['period']}분")
            print(f"   점수: {signal['score']}점")
        
        # 이벤트 유형별 통계
        event_counts = {
            'oi_changes': 0,
            'funding_changes': 0,
            'volume_spikes': 0
        }
        
        for signal in signals_to_analyze:
            event_counts[signal['event_type']] += 1
        
        print("\n" + "=" * 50)
        print(f"상위 {top_n}개 신호 이벤트 유형 분포")
        print("=" * 50)
        
        total = sum(event_counts.values())
        for event_type, count in event_counts.items():
            percentage = (count / total) * 100
            print(f"{event_type}: {count}개 ({percentage:.1f}%)")
        
        # 타임프레임별 통계
        timeframe_counts = {}
        for signal in signals_to_analyze:
            tf = signal['timeframe']
            if tf not in timeframe_counts:
                timeframe_counts[tf] = 0
            timeframe_counts[tf] += 1
        
        print("\n" + "=" * 50)
        print(f"상위 {top_n}개 신호 타임프레임 분포")
        print("=" * 50)
        
        for tf, count in sorted(timeframe_counts.items()):
            percentage = (count / total) * 100
            print(f"{tf}: {count}개 ({percentage:.1f}%)")
        
        # 기간별 통계
        period_counts = {}
        for signal in signals_to_analyze:
            period = signal['period']
            if period not in period_counts:
                period_counts[period] = 0
            period_counts[period] += 1
        
        print("\n" + "=" * 50)
        print(f"상위 {top_n}개 신호 기간 분포")
        print("=" * 50)
        
        for period, count in sorted(period_counts.items()):
            percentage = (count / total) * 100
            print(f"{period}분: {count}개 ({percentage:.1f}%)")
        
        # 심볼별 통계
        symbol_counts = {}
        for signal in signals_to_analyze:
            symbol = signal['symbol']
            if symbol not in symbol_counts:
                symbol_counts[symbol] = 0
            symbol_counts[symbol] += 1
        
        print("\n" + "=" * 50)
        print(f"상위 {top_n}개 신호 심볼 분포 (상위 10개)")
        print("=" * 50)
        
        # 심볼 카운트 기준 내림차순 정렬 후 상위 10개만 출력
        sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
        for symbol, count in sorted_symbols[:10]:
            percentage = (count / total) * 100
            print(f"{symbol}: {count}개 ({percentage:.1f}%)")
        
        # 100점 만점 신호 분석
        print("\n" + "=" * 50)
        print(f"100점 만점 신호 ({len(self.perfect_signals)}개)")
        print("=" * 50)
        
        for i, signal in enumerate(self.perfect_signals[:20], 1):  # 최대 20개까지만 출력
            print(f"\n{i}. 심볼: {signal['symbol']}")
            print(f"   타임프레임: {signal['timeframe']}")
            print(f"   이벤트 유형: {signal['event_type']}")
            print(f"   기간: {signal['period']}분")
        
        if len(self.perfect_signals) > 20:
            print(f"\n... 외 {len(self.perfect_signals) - 20}개 더 있음")
    
    def run_alarm_system(self, interval_minutes: int = 60, top_n: int = 20) -> None:
        """알람 시스템 실행 (주기적으로 알람 생성)"""
        logger.info(f"암호화폐 알람 시스템 시작 (간격: {interval_minutes}분, 상위 {top_n}개 알람)")
        
        try:
            while True:
                start_time = time.time()
                
                logger.info("모든 심볼 분석 시작...")
                self.analyze_all_symbols()
                
                logger.info("알람 생성 중...")
                alarms = self.generate_alarms(top_n)
                
                # 결과 저장
                self.save_results()
                
                # 상위 알람 출력
                if alarms:
                    logger.info(f"상위 알람 ({len(alarms)}개):")
                    for i, alarm in enumerate(alarms[:5], 1):
                        logger.info(f"{i}. {alarm['symbol']} {alarm['timeframe']} {alarm['event_type']} "
                                   f"{alarm['period']}분 - 점수: {alarm['score']}")
                    
                    if len(alarms) > 5:
                        logger.info(f"... 외 {len(alarms) - 5}개 더 있음")
                else:
                    logger.warning("생성된 알람이 없습니다.")
                
                # 다음 실행 시간 계산
                elapsed = time.time() - start_time
                sleep_time = max(0, interval_minutes * 60 - elapsed)
                
                logger.info(f"다음 분석까지 {sleep_time:.1f}초 대기 중...")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("사용자에 의해 알람 시스템이 중지되었습니다.")
        except Exception as e:
            logger.error(f"알람 시스템 실행 중 오류 발생: {e}")
            raise

# 메인 실행 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='암호화폐 알람 시스템')
    parser.add_argument('--data-dir', type=str, default='data/live', help='CSV 데이터 디렉토리')
    parser.add_argument('--results-dir', type=str, default='data/results', help='JSON 결과 디렉토리')
    parser.add_argument('--output-dir', type=str, default='data/alarms', help='알람 출력 디렉토리')
    parser.add_argument('--min-score', type=float, default=70.0, help='최소 가능성 점수')
    parser.add_argument('--interval', type=int, default=60, help='알람 생성 간격 (분)')
    parser.add_argument('--top-n', type=int, default=20, help='상위 N개 알람 생성')
    parser.add_argument('--symbol', type=str, help='특정 심볼만 분석')
    parser.add_argument('--analyze-only', action='store_true', help='분석만 수행 (알람 시스템 실행 안 함)')
    
    args = parser.parse_args()
    
    # 알람 시스템 초기화
    alarm_system = CryptoAlarmSystem(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        min_score=args.min_score
    )
    
    if args.symbol:
        # 특정 심볼만 분석
        alarm_system.print_symbol_analysis(args.symbol)
    elif args.analyze_only:
        # 분석만 수행
        alarm_system.analyze_all_symbols()
        alarm_system.print_top_signals_analysis()
        alarm_system.save_results()
    else:
        # 알람 시스템 실행
        alarm_system.run_alarm_system(args.interval, args.top_n) 