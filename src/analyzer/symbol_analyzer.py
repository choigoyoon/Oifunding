import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SymbolAnalyzer:
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
        with open("data/config/default.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
            
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
                        "oi_change": current['oi_change'],
                        "volume_ratio": current['volume_ratio'],
                        "trend_strength": current['trend_strength']
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
                        "funding_rate": current['funding_rate'],
                        "funding_prediction_gap": current['funding_prediction_gap'],
                        "volume_ratio": current['volume_ratio']
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
                        "oi_change": current['oi_change'],
                        "funding_rate": current['funding_rate'],
                        "volume_ratio": current['volume_ratio'],
                        "long_short_ratio": current['long_short_ratio'],
                        "liquidation_ratio": current['liquidation_ratio']
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
            # 결과 파일 저장
            output_file = self.data_dir / f"{self.symbol}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": self.symbol,
                    "results": results
                }, f, indent=2)
                
            logger.info(f"{self.symbol} 분석 결과 저장 완료")
            
        except Exception as e:
            logger.error(f"{self.symbol} 결과 저장 중 오류: {str(e)}") 