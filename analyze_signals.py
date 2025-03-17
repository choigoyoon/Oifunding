#!/usr/bin/env python
"""
추출된 신호의 분포를 분석하는 스크립트
- 이벤트 유형별 분포
- 타임프레임별 분포
- 점수 구간별 분포
- 심볼별 신호 수 분포
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analyze_signals.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AnalyzeSignals")

class SignalAnalyzer:
    """신호 분석기"""
    
    def __init__(self, signals_file: str = 'data/alarms/all_symbols_signals_70.0.json'):
        """
        초기화
        
        Args:
            signals_file: 신호 파일 경로
        """
        self.signals_file = Path(signals_file)
        self.signals = []
        self.total_signals = 0
        self.symbols_with_signals = 0
        self.symbols_without_signals = 0
    
    def load_signals(self):
        """신호 로드"""
        try:
            if not self.signals_file.exists():
                logger.error(f"신호 파일이 없습니다: {self.signals_file}")
                return False
            
            with open(self.signals_file, 'r') as f:
                data = json.load(f)
            
            self.signals = data.get('signals', [])
            self.total_signals = data.get('total_signals', 0)
            self.symbols_with_signals = data.get('symbols_with_signals', 0)
            self.symbols_without_signals = data.get('symbols_without_signals', 0)
            
            logger.info(f"총 {self.total_signals}개 신호 로드 완료")
            logger.info(f"신호가 있는 심볼: {self.symbols_with_signals}개")
            logger.info(f"신호가 없는 심볼: {self.symbols_without_signals}개")
            
            return True
            
        except Exception as e:
            logger.error(f"신호 로드 중 오류 발생: {e}")
            return False
    
    def analyze_event_types(self):
        """이벤트 유형별 분포 분석"""
        try:
            event_types = Counter()
            
            for signal in self.signals:
                event_type = signal.get('event_type', '')
                event_types[event_type] += 1
            
            logger.info("이벤트 유형별 분포:")
            for event_type, count in event_types.items():
                percentage = (count / self.total_signals) * 100
                logger.info(f"  {event_type}: {count}개 ({percentage:.1f}%)")
            
            return event_types
            
        except Exception as e:
            logger.error(f"이벤트 유형별 분포 분석 중 오류 발생: {e}")
            return {}
    
    def analyze_timeframes(self):
        """타임프레임별 분포 분석"""
        try:
            timeframes = Counter()
            
            for signal in self.signals:
                timeframe = signal.get('timeframe', '')
                timeframes[timeframe] += 1
            
            logger.info("타임프레임별 분포:")
            for timeframe, count in timeframes.items():
                percentage = (count / self.total_signals) * 100
                logger.info(f"  {timeframe}: {count}개 ({percentage:.1f}%)")
            
            return timeframes
            
        except Exception as e:
            logger.error(f"타임프레임별 분포 분석 중 오류 발생: {e}")
            return {}
    
    def analyze_periods(self):
        """기간별 분포 분석"""
        try:
            periods = Counter()
            
            for signal in self.signals:
                period = signal.get('period', '')
                periods[period] += 1
            
            logger.info("기간별 분포:")
            for period, count in sorted(periods.items(), key=lambda x: (isinstance(x[0], int), x[0])):
                percentage = (count / self.total_signals) * 100
                logger.info(f"  {period}분: {count}개 ({percentage:.1f}%)")
            
            return periods
            
        except Exception as e:
            logger.error(f"기간별 분포 분석 중 오류 발생: {e}")
            return {}
    
    def analyze_score_ranges(self):
        """점수 구간별 분포 분석"""
        try:
            score_ranges = {
                '90-100': 0,
                '80-90': 0,
                '70-80': 0,
                '60-70': 0,
                '50-60': 0,
                '40-50': 0,
                '30-40': 0,
                '20-30': 0,
                '10-20': 0,
                '0-10': 0
            }
            
            for signal in self.signals:
                score = signal.get('score', 0)
                
                if 90 <= score <= 100:
                    score_ranges['90-100'] += 1
                elif 80 <= score < 90:
                    score_ranges['80-90'] += 1
                elif 70 <= score < 80:
                    score_ranges['70-80'] += 1
                elif 60 <= score < 70:
                    score_ranges['60-70'] += 1
                elif 50 <= score < 60:
                    score_ranges['50-60'] += 1
                elif 40 <= score < 50:
                    score_ranges['40-50'] += 1
                elif 30 <= score < 40:
                    score_ranges['30-40'] += 1
                elif 20 <= score < 30:
                    score_ranges['20-30'] += 1
                elif 10 <= score < 20:
                    score_ranges['10-20'] += 1
                else:
                    score_ranges['0-10'] += 1
            
            logger.info("점수 구간별 분포:")
            for score_range, count in score_ranges.items():
                if count > 0:
                    percentage = (count / self.total_signals) * 100
                    logger.info(f"  {score_range} 점: {count}개 ({percentage:.1f}%)")
            
            return score_ranges
            
        except Exception as e:
            logger.error(f"점수 구간별 분포 분석 중 오류 발생: {e}")
            return {}
    
    def analyze_symbols(self):
        """심볼별 신호 수 분포 분석"""
        try:
            symbol_counts = defaultdict(int)
            
            for signal in self.signals:
                symbol = signal.get('symbol', '')
                symbol_counts[symbol] += 1
            
            # 신호 수 기준 내림차순 정렬
            sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
            
            logger.info(f"심볼별 신호 수 분포 (상위 10개):")
            for symbol, count in sorted_symbols[:10]:
                percentage = (count / self.total_signals) * 100
                logger.info(f"  {symbol}: {count}개 ({percentage:.1f}%)")
            
            # 신호 수 분포 통계
            signal_count_stats = Counter()
            for symbol, count in symbol_counts.items():
                signal_count_stats[count] += 1
            
            logger.info("심볼별 신호 수 통계:")
            for count, num_symbols in sorted(signal_count_stats.items()):
                percentage = (num_symbols / self.symbols_with_signals) * 100
                logger.info(f"  {count}개 신호를 가진 심볼: {num_symbols}개 ({percentage:.1f}%)")
            
            return sorted_symbols
            
        except Exception as e:
            logger.error(f"심볼별 신호 수 분포 분석 중 오류 발생: {e}")
            return []
    
    def analyze_lowest_score_symbols(self):
        """가장 낮은 점수의 심볼 분석"""
        try:
            symbol_min_scores = {}
            
            for signal in self.signals:
                symbol = signal.get('symbol', '')
                score = signal.get('score', 0)
                
                if symbol not in symbol_min_scores or score < symbol_min_scores[symbol]:
                    symbol_min_scores[symbol] = score
            
            # 점수 기준 오름차순 정렬
            sorted_symbols = sorted(symbol_min_scores.items(), key=lambda x: x[1])
            
            logger.info(f"가장 낮은 점수의 심볼 (하위 10개):")
            for symbol, score in sorted_symbols[:10]:
                logger.info(f"  {symbol}: {score}점")
            
            return sorted_symbols
            
        except Exception as e:
            logger.error(f"가장 낮은 점수의 심볼 분석 중 오류 발생: {e}")
            return []
    
    def run_analysis(self):
        """분석 실행"""
        if not self.load_signals():
            return False
        
        self.analyze_event_types()
        self.analyze_timeframes()
        self.analyze_periods()
        self.analyze_score_ranges()
        self.analyze_symbols()
        self.analyze_lowest_score_symbols()
        
        return True

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='신호 분포 분석')
    parser.add_argument('--file', type=str, default='data/alarms/all_symbols_signals_70.0.json', help='신호 파일 경로')
    
    args = parser.parse_args()
    
    # 신호 분석기 초기화
    analyzer = SignalAnalyzer(signals_file=args.file)
    
    # 분석 실행
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 