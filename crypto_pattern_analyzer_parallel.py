"""
암호화폐 데이터 분석 - 중요 변동점 감지 시스템
- OI, 거래량, 펀딩비의 중요 변동점 감지
- 심볼별 분석 결과를 JSON으로 저장
- 병렬 처리를 통한 성능 최적화
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import os
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_pattern_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoPatternAnalyzer")

class SymbolAnalyzer:
    """단일 심볼 분석기 - 중요 변동점 감지"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.data_dir = Path("data/live")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.load_config()
        self.df = None
        
    def load_config(self) -> dict:
        """심볼별 설정 로드"""
        config_path = self.data_dir / f"{self.symbol}_config.json"
        
        # 기본 설정 로드
        try:
            with open("data/config/default.json", 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            # 기본 설정 파일이 없으면 기본값 사용
            config = {
                "min_data_points": 20,
                "oi_change_threshold": 0.05,
                "funding_threshold": 0.001,
                "volume_ratio_threshold": 3.0
            }
            
        # 심볼별 설정이 있으면 오버라이드
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                symbol_config = json.load(f)
                config.update(symbol_config)
                
        return config
    
    def load_data(self) -> bool:
        """데이터 로드"""
        try:
            data_path = self.data_dir / f"{self.symbol}.csv"
            if not data_path.exists():
                logger.error(f"{self.symbol} 데이터 파일이 없습니다")
                return False
                
            # 데이터 로드
            df = pd.read_csv(data_path)
            
            # datetime 처리
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # 컬럼 타입 변환
            numeric_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'open_interest', 'funding_rate', 'predicted_funding_rate',
                'liquidation', 'long_short_ratio'
            ]
            
            for col in numeric_columns:
                try:
                    if col in df.columns:
                        # 문자열 전처리
                        if df[col].dtype == object:
                            df[col] = df[col].astype(str).str.extract('(\d+\.?\d*)', expand=False)
                        # 숫자로 변환
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"{self.symbol} {col} 컬럼 변환 중 오류: {str(e)}")
                    
            # 결측치 처리
            df = df.fillna(method='ffill')
            
            # 데이터 정렬
            df = df.sort_values('datetime')
            
            self.df = df
            logger.info(f"{self.symbol} 데이터 로드 완료: {len(df)}행")
            return True
            
        except Exception as e:
            logger.error(f"{self.symbol} 데이터 로드 중 오류: {str(e)}")
            return False
    
    def prepare_timeframes(self) -> Dict[str, pd.DataFrame]:
        """멀티 타임프레임 데이터 준비"""
        if self.df is None:
            return {}
            
        timeframes = {
            '5M': 1,    # 5분 = 기본 데이터
            '30M': 6,   # 30분 = 6캔들
            '1H': 12,   # 1시간 = 12캔들
            '4H': 48    # 4시간 = 48캔들
        }
        
        tf_data = {}
        
        for tf, period in timeframes.items():
            try:
                # 리샘플링
                resampled = self.df.resample(f'{period*5}min', on='datetime', label='right').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'open_interest': 'last',
                    'funding_rate': 'last',
                    'predicted_funding_rate': 'last',
                    'liquidation': 'sum',
                    'long_short_ratio': 'last'
                })
                
                # 지표 계산
                df_tf = resampled.copy()
                
                # OI 관련 지표
                df_tf['oi_change'] = df_tf['open_interest'].pct_change(fill_method=None)
                df_tf['oi_ma'] = df_tf['open_interest'].rolling(window=20, min_periods=1).mean()
                df_tf['oi_std'] = df_tf['open_interest'].rolling(window=20, min_periods=1).std()
                
                # 펀딩비 관련 지표
                df_tf['funding_sign_change'] = np.sign(df_tf['funding_rate']) != np.sign(df_tf['funding_rate'].shift(1))
                df_tf['funding_prediction_gap'] = df_tf['funding_rate'] - df_tf['predicted_funding_rate']
                df_tf['funding_ma'] = df_tf['funding_rate'].rolling(window=20, min_periods=1).mean()
                df_tf['funding_std'] = df_tf['funding_rate'].rolling(window=20, min_periods=1).std()
                
                # 거래량 관련 지표
                df_tf['volume_ma'] = df_tf['volume'].rolling(window=20, min_periods=1).mean()
                df_tf['volume_ratio'] = df_tf['volume'] / df_tf['volume_ma']
                df_tf['volume_std'] = df_tf['volume'].rolling(window=20, min_periods=1).std()
                
                # 가격 변동성
                df_tf['returns'] = df_tf['close'].pct_change(fill_method=None)
                df_tf['volatility'] = df_tf['returns'].rolling(window=20, min_periods=1).std()
                df_tf['volatility_ma'] = df_tf['volatility'].rolling(window=20, min_periods=1).mean()
                
                # 청산/롱숏 관련
                df_tf['liquidation_ma'] = df_tf['liquidation'].rolling(window=20, min_periods=1).mean()
                df_tf['liquidation_ratio'] = df_tf['liquidation'] / df_tf['liquidation_ma']
                df_tf['ls_extreme'] = (df_tf['long_short_ratio'] > 1.25) | (df_tf['long_short_ratio'] < 0.75)
                
                # 추세 강도
                df_tf['atr'] = self.calculate_atr(df_tf)
                df_tf['trend_strength'] = abs(df_tf['close'] - df_tf['close'].shift(10)) / (df_tf['atr'] * 10)
                
                # NaN 값 처리
                df_tf.fillna(0, inplace=True)
                
                tf_data[tf] = df_tf
                
            except Exception as e:
                logger.error(f"{self.symbol} {tf} 타임프레임 준비 중 오류: {str(e)}")
        
        return tf_data
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR(Average True Range) 계산"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def analyze(self) -> Dict:
        """심볼 분석 실행"""
        try:
            # 1. 데이터 로드
            if not self.load_data():
                return {}
                
            # 2. 멀티 타임프레임 데이터 준비
            tf_data = self.prepare_timeframes()
            if not tf_data:
                return {}
                
            # 3. 각 타임프레임별 분석
            results = {}
            for tf, df in tf_data.items():
                tf_result = self.analyze_timeframe(tf, df)
                if tf_result:
                    results[tf] = tf_result
                    
            # 4. 결과 저장
            self.save_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"{self.symbol} 분석 중 오류: {str(e)}")
            return {}
    
    def analyze_timeframe(self, timeframe: str, df: pd.DataFrame) -> Dict:
        """단일 타임프레임 분석"""
        try:
            result = {
                'timestamp': df.index[-1].isoformat(),
                'patterns': []
            }
            
            # 최소 20개의 데이터 필요
            if len(df) < 20:
                return result
                
            # 현재 데이터
            current = df.iloc[-1]
            
            # 1. OI 기반 패턴
            oi_pattern = self.detect_oi_pattern(df)
            if oi_pattern:
                result['patterns'].append(oi_pattern)
                
            # 2. 펀딩비 기반 패턴
            funding_pattern = self.detect_funding_pattern(df)
            if funding_pattern:
                result['patterns'].append(funding_pattern)
                
            # 3. 복합 패턴
            complex_pattern = self.detect_complex_pattern(df)
            if complex_pattern:
                result['patterns'].append(complex_pattern)
                
            return result
            
        except Exception as e:
            logger.error(f"{self.symbol} {timeframe} 분석 중 오류: {str(e)}")
            return {}
    
    def detect_oi_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """OI 기반 패턴 감지"""
        try:
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # OI 급증 후 하락
            oi_surge_drop = (
                df['oi_change'].iloc[-4:-1].mean() > 0.07 and 
                current['oi_change'] < -0.01
            )
            
            # OI 급락 후 반등
            oi_drop_bounce = (
                df['oi_change'].iloc[-4:-1].mean() < -0.10 and
                current['oi_change'] > 0.01
            )
            
            # OI 횡보 후 급변
            lookback = 20
            oi_breakout = (
                abs(df['oi_change'].iloc[-lookback:-1].max()) <= 0.03 and
                abs(current['oi_change']) > 0.05
            )
            
            if any([oi_surge_drop, oi_drop_bounce, oi_breakout]):
                pattern_type = ("OI_SURGE_DROP" if oi_surge_drop else
                              "OI_DROP_BOUNCE" if oi_drop_bounce else
                              "OI_BREAKOUT")
                              
                return {
                    "type": pattern_type,
                    "confidence": self.calculate_confidence(current),
                    "metrics": {
                        "oi_change": float(current['oi_change']),
                        "volume_ratio": float(current['volume_ratio']),
                        "trend_strength": float(current['trend_strength'])
                    }
                }
                
            return None
            
        except Exception as e:
            logger.error(f"{self.symbol} OI 패턴 감지 중 오류: {str(e)}")
            return None
    
    def detect_funding_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """펀딩비 기반 패턴 감지"""
        try:
            current = df.iloc[-1]
            
            # 펀딩비 방향 전환
            funding_flip = current['funding_sign_change']
            
            # 극단적 펀딩비
            funding_extreme = abs(current['funding_rate']) > 0.001
            
            # 예측값과 실제값 차이
            funding_divergence = abs(current['funding_prediction_gap']) > 0.0005
            
            if any([funding_flip, funding_extreme, funding_divergence]):
                pattern_type = ("FUNDING_FLIP" if funding_flip else
                              "FUNDING_EXTREME" if funding_extreme else
                              "FUNDING_DIVERGENCE")
                              
                return {
                    "type": pattern_type,
                    "confidence": self.calculate_confidence(current),
                    "metrics": {
                        "funding_rate": float(current['funding_rate']),
                        "funding_prediction_gap": float(current['funding_prediction_gap']),
                        "volume_ratio": float(current['volume_ratio'])
                    }
                }
                
            return None
            
        except Exception as e:
            logger.error(f"{self.symbol} 펀딩비 패턴 감지 중 오류: {str(e)}")
            return None
    
    def detect_complex_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """복합 패턴 감지"""
        try:
            current = df.iloc[-1]
            
            # OI와 펀딩비 다이버전스
            oi_funding_divergence = (
                (current['oi_change'] > 0.08 and current['funding_rate'] < -0.0005) or
                (current['oi_change'] < -0.08 and current['funding_rate'] > 0.0005)
            )
            
            # 거래량 급증 + 롱숏 극단
            volume_ls_extreme = (
                current['volume_ratio'] > 5.0 and
                abs(current['long_short_ratio'] - 1) > 0.25
            )
            
            # 대량 청산
            liquidation_surge = current['liquidation_ratio'] > 3.0
            
            if any([oi_funding_divergence, volume_ls_extreme, liquidation_surge]):
                pattern_type = ("OI_FUNDING_DIVERGENCE" if oi_funding_divergence else
                              "VOLUME_LS_EXTREME" if volume_ls_extreme else
                              "LIQUIDATION_SURGE")
                              
                return {
                    "type": pattern_type,
                    "confidence": self.calculate_confidence(current),
                    "metrics": {
                        "oi_change": float(current['oi_change']),
                        "funding_rate": float(current['funding_rate']),
                        "volume_ratio": float(current['volume_ratio']),
                        "long_short_ratio": float(current['long_short_ratio']),
                        "liquidation_ratio": float(current['liquidation_ratio'])
                    }
                }
                
            return None
            
        except Exception as e:
            logger.error(f"{self.symbol} 복합 패턴 감지 중 오류: {str(e)}")
            return None
    
    def calculate_confidence(self, current: pd.Series) -> float:
        """신뢰도 점수 계산"""
        confidence = 60  # 기본 점수
        
        # 보조 지표 가점
        if current['volume_ratio'] > 3.0: confidence += 10
        if current['ls_extreme']: confidence += 10
        if current['liquidation_ratio'] > 2.0: confidence += 10
        if current['trend_strength'] > 1.0: confidence += 5
        
        return min(100, confidence)
    
    def save_results(self, results: Dict):
        """분석 결과 저장"""
        try:
            # 결과 디렉토리 생성
            output_dir = Path("data/results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 결과 파일 저장
            output_file = output_dir / f"{self.symbol}_changes.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": self.symbol,
                    "results": results
                }, f, indent=2)
                
            logger.info(f"{self.symbol} 분석 결과 저장 완료")
            
        except Exception as e:
            logger.error(f"{self.symbol} 결과 저장 중 오류: {str(e)}")


class AnalysisWorker:
    """여러 심볼에 대한 분석 작업 관리"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.results = {}
        
    def get_symbols(self) -> List[str]:
        """분석할 심볼 목록 가져오기"""
        try:
            live_dir = Path("data/live")
            return [f.stem for f in live_dir.glob("*.csv")]
        except Exception as e:
            logger.error(f"심볼 목록 로드 중 오류: {str(e)}")
            return []
            
    def analyze_symbol(self, symbol: str) -> Dict:
        """단일 심볼 분석"""
        try:
            analyzer = SymbolAnalyzer(symbol)
            results = analyzer.analyze()
            logger.info(f"{symbol} 분석 완료")
            return {symbol: results}
        except Exception as e:
            logger.error(f"{symbol} 분석 중 오류: {str(e)}")
            return {symbol: {}}
            
    def run(self):
        """병렬 분석 실행"""
        try:
            # 1. 심볼 목록 가져오기
            symbols = self.get_symbols()
            if not symbols:
                logger.error("분석할 심볼이 없습니다.")
                return
                
            logger.info(f"총 {len(symbols)}개 심볼 분석 시작")
            start_time = datetime.now()
            
            # 2. 병렬 처리
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.analyze_symbol, symbol): symbol 
                    for symbol in symbols
                }
                
                # 결과 수집
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        self.results.update(result)
                    except Exception as e:
                        logger.error(f"{symbol} 처리 중 오류: {str(e)}")
            
            # 실행 시간 계산
            duration = datetime.now() - start_time
            logger.info(f"전체 분석 완료: {len(symbols)}개 심볼, 소요시간: {duration}")
            
            # 결과 요약 저장
            self.save_summary()
            
        except Exception as e:
            logger.error(f"분석 실행 중 오류: {str(e)}")
    
    def save_summary(self):
        """분석 결과 요약 저장"""
        try:
            # 결과 디렉토리 생성
            output_dir = Path("data/summary")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 패턴 통계 계산
            pattern_stats = {
                "total_symbols": len(self.results),
                "symbols_with_patterns": 0,
                "pattern_counts": {
                    "oi_patterns": 0,
                    "funding_patterns": 0,
                    "complex_patterns": 0
                },
                "pattern_types": {}
            }
            
            # 각 심볼별 패턴 집계
            for symbol, result in self.results.items():
                if not result:
                    continue
                
                has_patterns = False
                for tf, tf_result in result.items():
                    if 'patterns' in tf_result and tf_result['patterns']:
                        has_patterns = True
                        for pattern in tf_result['patterns']:
                            pattern_type = pattern['type']
                            
                            # 패턴 유형별 카운트
                            if pattern_type.startswith('OI_'):
                                pattern_stats["pattern_counts"]["oi_patterns"] += 1
                            elif pattern_type.startswith('FUNDING_'):
                                pattern_stats["pattern_counts"]["funding_patterns"] += 1
                            else:
                                pattern_stats["pattern_counts"]["complex_patterns"] += 1
                            
                            # 세부 패턴 유형별 카운트
                            if pattern_type not in pattern_stats["pattern_types"]:
                                pattern_stats["pattern_types"][pattern_type] = 0
                            pattern_stats["pattern_types"][pattern_type] += 1
                
                if has_patterns:
                    pattern_stats["symbols_with_patterns"] += 1
            
            # 요약 파일 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = output_dir / f"analysis_summary_{timestamp}.json"
            
            with open(summary_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "stats": pattern_stats
                }, f, indent=2)
                
            logger.info(f"분석 요약 저장 완료: {summary_file}")
            
        except Exception as e:
            logger.error(f"분석 요약 저장 중 오류: {str(e)}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='암호화폐 패턴 분석 시스템')
    parser.add_argument('--workers', type=int, default=4, help='병렬 작업자 수')
    parser.add_argument('--symbol', type=str, help='특정 심볼만 분석')
    
    args = parser.parse_args()
    
    if args.symbol:
        # 단일 심볼 분석
        logger.info(f"{args.symbol} 심볼 분석 시작")
        analyzer = SymbolAnalyzer(args.symbol)
        results = analyzer.analyze()
        if results:
            logger.info(f"{args.symbol} 분석 완료: {sum(len(tf_result.get('patterns', [])) for tf_result in results.values())}개 패턴 발견")
        else:
            logger.warning(f"{args.symbol} 분석 결과 없음")
    else:
        # 모든 심볼 병렬 분석
        worker = AnalysisWorker(max_workers=args.workers)
        worker.run()


if __name__ == "__main__":
    main() 