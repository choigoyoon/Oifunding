#!/usr/bin/env python
"""
모든 심볼에서 최소 1개 이상의 신호를 추출하는 스크립트
- 적응형 임계값 접근법 사용
- 각 심볼마다 최소 1개 이상의 신호 보장
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extract_all_symbols_signals.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ExtractAllSymbolsSignals")

class SignalExtractor:
    """모든 심볼에서 신호 추출"""
    
    def __init__(self, 
                 results_dir: str = 'data/results',
                 alarms_dir: str = 'data/alarms',
                 initial_min_score: float = 70.0,
                 min_score_step: float = 5.0):
        """
        초기화
        
        Args:
            results_dir: 결과 디렉토리
            alarms_dir: 알람 디렉토리
            initial_min_score: 초기 최소 점수
            min_score_step: 점수 감소 단계
        """
        self.results_dir = Path(results_dir)
        self.alarms_dir = Path(alarms_dir)
        self.initial_min_score = initial_min_score
        self.min_score_step = min_score_step
        
        # 결과 저장 변수
        self.all_signals = {}
        self.symbols_with_signals = set()
        self.symbols_without_signals = set()
        
        # 디렉토리 생성
        self.alarms_dir.mkdir(parents=True, exist_ok=True)
    
    def load_analysis_results(self):
        """분석 결과 로드"""
        try:
            # 승률 기반 신호 분석 결과 로드
            signal_file = self.alarms_dir / "win_rate_analysis.json"
            if not signal_file.exists():
                logger.warning("신호 분석 결과 파일이 없습니다")
                return False
            
            with open(signal_file, 'r') as f:
                signal_data = json.load(f)
            
            # 모든 심볼 목록 추출
            all_symbols = set(signal_data.get('symbols', {}).keys())
            
            if not all_symbols:
                logger.warning("분석된 심볼이 없습니다")
                return False
            
            logger.info(f"총 {len(all_symbols)}개 심볼 로드 완료")
            
            # 모든 심볼에 대한 신호 추출
            self.extract_signals_with_adaptive_threshold(signal_data, all_symbols)
            
            return True
            
        except Exception as e:
            logger.error(f"분석 결과 로드 중 오류 발생: {e}")
            return False
    
    def extract_signals_with_adaptive_threshold(self, signal_data, all_symbols):
        """적응형 임계값으로 모든 심볼에서 신호 추출"""
        # 초기 임계값 설정
        min_score = self.initial_min_score
        
        # 모든 심볼에 대한 신호 추출
        self.all_signals = {}
        self.symbols_with_signals = set()
        self.symbols_without_signals = set()
        
        # 첫 번째 패스: 초기 임계값으로 신호 추출
        for symbol, data in signal_data.get('symbols', {}).items():
            best_signals = []
            
            for tf, events in data.get('probability_scores', {}).items():
                if tf in ['oi_changes', 'funding_changes', 'volume_spikes']:
                    continue
                
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
            
            if best_signals:
                self.all_signals[symbol] = best_signals
                self.symbols_with_signals.add(symbol)
            else:
                self.symbols_without_signals.add(symbol)
        
        logger.info(f"초기 임계값 {min_score}점으로 {len(self.symbols_with_signals)}개 심볼에서 신호 추출")
        logger.info(f"신호가 없는 심볼: {len(self.symbols_without_signals)}개")
        
        # 두 번째 패스: 신호가 없는 심볼에 대해 임계값 낮추기
        while self.symbols_without_signals and min_score > 30.0:  # 최소 30점까지만 낮춤
            min_score -= self.min_score_step
            logger.info(f"임계값을 {min_score}점으로 낮춰서 다시 시도")
            
            symbols_still_without_signals = set()
            
            for symbol in self.symbols_without_signals:
                if symbol not in signal_data.get('symbols', {}):
                    symbols_still_without_signals.add(symbol)
                    continue
                
                data = signal_data['symbols'][symbol]
                best_signals = []
                
                for tf, events in data.get('probability_scores', {}).items():
                    if tf in ['oi_changes', 'funding_changes', 'volume_spikes']:
                        continue
                    
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
                
                if best_signals:
                    self.all_signals[symbol] = best_signals
                    self.symbols_with_signals.add(symbol)
                else:
                    symbols_still_without_signals.add(symbol)
            
            self.symbols_without_signals = symbols_still_without_signals
            logger.info(f"임계값 {min_score}점으로 {len(self.symbols_with_signals)}개 심볼에서 신호 추출")
            logger.info(f"신호가 없는 심볼: {len(self.symbols_without_signals)}개")
        
        # 세 번째 패스: 여전히 신호가 없는 심볼에 대해 가장 높은 점수의 신호 추출
        if self.symbols_without_signals:
            logger.info(f"남은 {len(self.symbols_without_signals)}개 심볼에 대해 가장 높은 점수의 신호 추출")
            
            for symbol in self.symbols_without_signals:
                if symbol not in signal_data.get('symbols', {}):
                    continue
                
                data = signal_data['symbols'][symbol]
                best_signals = []
                
                for tf, events in data.get('probability_scores', {}).items():
                    if tf in ['oi_changes', 'funding_changes', 'volume_spikes']:
                        continue
                    
                    for event_type, periods in events.items():
                        for period, score in periods.items():
                            best_signals.append({
                                'symbol': symbol,
                                'timeframe': tf,
                                'event_type': event_type,
                                'period': period,
                                'score': score
                            })
                
                # 점수 기준 내림차순 정렬
                best_signals.sort(key=lambda x: x['score'], reverse=True)
                
                if best_signals:
                    # 가장 높은 점수의 신호만 추출
                    self.all_signals[symbol] = [best_signals[0]]
                    self.symbols_with_signals.add(symbol)
                    self.symbols_without_signals.remove(symbol)
            
            logger.info(f"최종적으로 {len(self.symbols_with_signals)}개 심볼에서 신호 추출")
            logger.info(f"신호가 없는 심볼: {len(self.symbols_without_signals)}개")
    
    def save_results(self):
        """결과 저장"""
        try:
            # 모든 심볼의 신호 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.alarms_dir / f"all_symbols_signals_{self.initial_min_score}.json"
            
            # 모든 신호 추출
            all_signals_flat = []
            for symbol, signals in self.all_signals.items():
                all_signals_flat.extend(signals)
            
            # 점수 기준 내림차순 정렬
            all_signals_flat.sort(key=lambda x: x['score'], reverse=True)
            
            # 결과 저장
            with open(output_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_signals': len(all_signals_flat),
                    'symbols_with_signals': len(self.symbols_with_signals),
                    'symbols_without_signals': len(self.symbols_without_signals),
                    'min_score_used': self.initial_min_score,
                    'signals': all_signals_flat
                }, f, indent=2)
            
            logger.info(f"모든 심볼의 신호가 {output_file}에 저장되었습니다.")
            
            # 심볼별 최고 점수 신호 저장
            best_signals_file = self.alarms_dir / f"all_symbols_best_signals_{timestamp}.json"
            best_signals = {}
            
            for symbol, signals in self.all_signals.items():
                if signals:
                    best_signals[symbol] = signals[0]  # 가장 높은 점수의 신호
            
            with open(best_signals_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_symbols': len(best_signals),
                    'best_signals': best_signals
                }, f, indent=2)
            
            logger.info(f"심볼별 최고 점수 신호가 {best_signals_file}에 저장되었습니다.")
            
            return True
            
        except Exception as e:
            logger.error(f"결과 저장 중 오류 발생: {e}")
            return False

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='모든 심볼에서 신호 추출')
    parser.add_argument('--min-score', type=float, default=70.0, help='초기 최소 점수')
    parser.add_argument('--step', type=float, default=5.0, help='점수 감소 단계')
    
    args = parser.parse_args()
    
    # 신호 추출기 초기화
    extractor = SignalExtractor(
        initial_min_score=args.min_score,
        min_score_step=args.step
    )
    
    # 분석 결과 로드 및 신호 추출
    if extractor.load_analysis_results():
        # 결과 저장
        extractor.save_results()
    else:
        logger.error("신호 추출 실패")

if __name__ == "__main__":
    main() 