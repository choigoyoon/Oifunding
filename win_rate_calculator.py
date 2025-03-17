import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple
import glob
from datetime import datetime, timedelta

class WinRateCalculator:
    """알람 승률 및 가능성 점수 계산기"""
    
    def __init__(self, data_dir: str = 'data/live', results_dir: str = 'data/results'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.win_rates = {}
        self.probability_scores = {}
    
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
    
    def calculate_win_rates(self, symbol: str, lookforward_periods: List[int] = [5, 15, 30, 60]) -> Dict:
        """알람 승률 계산
        
        Args:
            symbol: 분석할 심볼
            lookforward_periods: 미래 몇 분 동안의 가격 변화를 확인할지 (분 단위)
            
        Returns:
            각 알람 유형별 승률 정보
        """
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
                    df, oi_changes, 'oi', lookforward_periods
                )
            
            # 2. 자금 비율 변화 승률 계산
            funding_changes = json_data['results'][tf].get('funding_changes', [])
            if funding_changes:
                win_rates[tf]['funding_changes'] = self._calculate_event_win_rate(
                    df, funding_changes, 'funding', lookforward_periods
                )
            
            # 3. 거래량 스파이크 승률 계산
            volume_spikes = json_data['results'][tf].get('volume_spikes', [])
            if volume_spikes:
                win_rates[tf]['volume_spikes'] = self._calculate_event_win_rate(
                    df, volume_spikes, 'volume', lookforward_periods
                )
        
        self.win_rates[symbol] = win_rates
        return win_rates
    
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
                print(f"이벤트 처리 중 오류 발생: {e}")
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
    
    def get_best_signals(self, symbol: str, min_score: float = 70.0) -> List[Dict]:
        """가장 높은 가능성 점수를 가진 신호 추출"""
        if symbol not in self.probability_scores:
            self.calculate_probability_score(symbol)
        
        best_signals = []
        
        for tf, events in self.probability_scores[symbol].items():
            for event_type, periods in events.items():
                for period, score in periods.items():
                    if score >= min_score:
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
        
        for symbol in symbols:
            try:
                win_rates = self.calculate_win_rates(symbol)
                probability_scores = self.calculate_probability_score(symbol)
                best_signals = self.get_best_signals(symbol)
                
                all_results[symbol] = {
                    'win_rates': win_rates,
                    'probability_scores': probability_scores,
                    'best_signals': best_signals
                }
                
                print(f"{symbol} 분석 완료: 최고 점수 신호 {len(best_signals)}개 발견")
            except Exception as e:
                print(f"{symbol} 분석 중 오류 발생: {e}")
        
        return all_results
    
    def save_results(self, output_file: str = 'data/win_rate_analysis.json'):
        """분석 결과 저장"""
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {}
        }
        
        for symbol in self.win_rates.keys():
            all_results['symbols'][symbol] = {
                'win_rates': self.win_rates.get(symbol, {}),
                'probability_scores': self.probability_scores.get(symbol, {}),
                'best_signals': self.get_best_signals(symbol)
            }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"분석 결과가 {output_file}에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    calculator = WinRateCalculator()
    
    # 단일 심볼 분석
    # win_rates = calculator.calculate_win_rates("BTC")
    # probability_scores = calculator.calculate_probability_score("BTC")
    # best_signals = calculator.get_best_signals("BTC")
    # print(f"BTC 최고 점수 신호: {best_signals}")
    
    # 모든 심볼 분석
    all_results = calculator.analyze_all_symbols()
    calculator.save_results()
    
    # 상위 신호 출력
    all_signals = []
    for symbol, data in all_results.items():
        all_signals.extend(data['best_signals'])
    
    # 점수 기준 정렬
    all_signals.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n상위 10개 신호:")
    for i, signal in enumerate(all_signals[:10], 1):
        print(f"{i}. {signal['symbol']} {signal['timeframe']} {signal['event_type']} - 점수: {signal['score']}") 