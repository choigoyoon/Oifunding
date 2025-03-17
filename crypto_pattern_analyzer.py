"""
암호화폐 데이터 분석 - 중요 변동점 감지 시스템
- OI, 거래량, 펀딩비의 중요 변동점 감지
- 심볼별 분석 결과를 JSON으로 저장
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from scipy import signal

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SymbolAnalyzer:
    """단일 심볼 분석기 - 중요 변동점 감지"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.data_dir = Path("data/live")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.load_config()
        self.df = None
        self.timeframe_data = {}
        
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
            
            # 필수 컬럼 확인 및 생성
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            optional_columns = ['open_interest', 'funding_rate', 'predicted_funding_rate', 
                               'liquidation', 'long_short_ratio']
            
            # 필수 컬럼이 없으면 오류
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"{self.symbol} 필수 컬럼 {col}이 없습니다")
                    return False
            
            # 선택적 컬럼이 없으면 0으로 채움
            for col in optional_columns:
                if col not in df.columns:
                    logger.warning(f"{self.symbol} 선택적 컬럼 {col}이 없어 0으로 채웁니다")
                    df[col] = 0
            
            # 컬럼 타입 변환
            numeric_columns = required_columns + optional_columns
            
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
        """멀티 타임프레임 데이터 준비 (1분봉 기준)"""
        if self.df is None:
            return {}
            
        # 1분봉 기준으로 타임프레임 설정 업데이트
        timeframes = {
            '1M': 1,    # 1분 = 기본 데이터
            '5M': 5,    # 5분 = 5캔들
            '15M': 15,  # 15분 = 15캔들
            '30M': 30,  # 30분 = 30캔들
            '1H': 60,   # 1시간 = 60캔들
            '4H': 240   # 4시간 = 240캔들
        }
        
        tf_data = {}
        
        # 1분봉은 원본 데이터 사용
        tf_data['1M'] = self.df.copy()
        
        # 1분봉 데이터에 지표 추가 (open_interest가 있는 경우에만)
        if 'open_interest' in self.df.columns and self.df['open_interest'].sum() > 0:
            tf_data['1M']['oi_change'] = tf_data['1M']['open_interest'].pct_change()
        else:
            tf_data['1M']['oi_change'] = 0
        
        if 'funding_rate' in self.df.columns:
            tf_data['1M']['funding_prev'] = tf_data['1M']['funding_rate'].shift(1)
            tf_data['1M']['funding_sign_change'] = np.sign(tf_data['1M']['funding_rate']) != np.sign(tf_data['1M']['funding_prev'])
        else:
            tf_data['1M']['funding_prev'] = 0
            tf_data['1M']['funding_sign_change'] = False
        
        tf_data['1M'].dropna(inplace=True)
        
        # 다른 타임프레임은 리샘플링
        for tf, period in timeframes.items():
            if tf == '1M':  # 1분봉은 이미 처리함
                continue
                
            # 기본 집계 컬럼
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # 선택적 컬럼 추가
            if 'open_interest' in self.df.columns and self.df['open_interest'].sum() > 0:
                agg_dict['open_interest'] = 'last'
            
            if 'funding_rate' in self.df.columns:
                agg_dict['funding_rate'] = 'last'
            
            if 'predicted_funding_rate' in self.df.columns:
                agg_dict['predicted_funding_rate'] = 'last'
            
            if 'liquidation' in self.df.columns and self.df['liquidation'].sum() > 0:
                agg_dict['liquidation'] = 'sum'
            
            if 'long_short_ratio' in self.df.columns and self.df['long_short_ratio'].sum() > 0:
                agg_dict['long_short_ratio'] = 'last'
            
            # 리샘플링
            resampled = self.df.resample(f'{period}min', on='datetime', label='right').agg(agg_dict)
            
            # 지표 계산
            df_tf = resampled.copy()
            
            # OI 변화율 (있는 경우에만)
            if 'open_interest' in df_tf.columns and df_tf['open_interest'].sum() > 0:
                df_tf['oi_change'] = df_tf['open_interest'].pct_change()
            else:
                df_tf['oi_change'] = 0
                df_tf['open_interest'] = 0
            
            # 펀딩비 부호 변경 (있는 경우에만)
            if 'funding_rate' in df_tf.columns:
                df_tf['funding_prev'] = df_tf['funding_rate'].shift(1)
                df_tf['funding_sign_change'] = np.sign(df_tf['funding_rate']) != np.sign(df_tf['funding_prev'])
            else:
                df_tf['funding_rate'] = 0
                df_tf['funding_prev'] = 0
                df_tf['funding_sign_change'] = False
            
            # 없는 컬럼 추가
            for col in ['liquidation', 'long_short_ratio', 'predicted_funding_rate']:
                if col not in df_tf.columns:
                    df_tf[col] = 0
            
            # NaN 처리
            df_tf.dropna(inplace=True)
            
            if len(df_tf) > 0:
                tf_data[tf] = df_tf
                
        self.timeframe_data = tf_data
        return tf_data
            
    def find_significant_changes(self) -> Dict:
        """중요한 변동점 찾기"""
        if not self.timeframe_data:
            return {}
            
        results = {}
        
        for tf, df in self.timeframe_data.items():
            if len(df) < 20:  # 최소 데이터 요구
                continue
                
            tf_results = {
                'oi_changes': self._find_oi_changes(df),
                'funding_changes': self._find_funding_changes(df),
                'volume_spikes': self._find_volume_spikes(df)
            }
            
            results[tf] = tf_results
                
        return results
        
    def _find_oi_changes(self, df: pd.DataFrame) -> List[Dict]:
        """OI 중요 변동점 찾기"""
        changes = []
        
        # OI 데이터가 없거나 모두 0인 경우 빈 리스트 반환
        if 'open_interest' not in df.columns or df['open_interest'].sum() == 0:
            return changes
        
        # 이동평균 계산
        df['oi_ma'] = df['open_interest'].rolling(window=20).mean()
        df['oi_std'] = df['open_interest'].rolling(window=20).std()
        
        # 기준 OI 임계값 설정
        threshold = 0.05  # 5% 변화
        
        # 전체 데이터 순회하며 중요 변동점 찾기
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # 1. 급격한 변화율 (5% 이상)
            significant_change = abs(current['oi_change']) > threshold
            
            # 2. 방향 전환
            direction_change = (previous['oi_change'] > 0 and current['oi_change'] < 0) or \
                               (previous['oi_change'] < 0 and current['oi_change'] > 0)
            
            # 3. 이동평균으로부터 크게 벗어남 (2 표준편차 이상)
            deviation = False
            if not pd.isna(current['oi_std']) and current['oi_std'] > 0:
                deviation = abs(current['open_interest'] - current['oi_ma']) > 2 * current['oi_std']
            
            if significant_change or (direction_change and abs(current['oi_change']) > threshold/2) or deviation:
                # timestamp 처리 - numpy.int64 타입 처리
                timestamp = current.name
                if hasattr(timestamp, 'isoformat'):
                    timestamp = timestamp.isoformat()
                else:
                    timestamp = str(timestamp)
                    
                changes.append({
                    'timestamp': timestamp,
                    'price': float(current['close']),
                    'oi': float(current['open_interest']),
                    'oi_change': float(current['oi_change']),
                    'type': 'significant_change' if significant_change else \
                           'direction_change' if direction_change else 'deviation',
                    'direction': 'up' if current['oi_change'] > 0 else 'down'
                })
        
        # 중요도 순으로 정렬 (변화율 크기)
        changes.sort(key=lambda x: abs(float(x['oi_change'])), reverse=True)
        
        # 상위 10개만 반환 (너무 많은 지점을 반환하지 않도록)
        return changes[:10]
        
    def _find_funding_changes(self, df: pd.DataFrame) -> List[Dict]:
        """펀딩비 중요 변동점 찾기"""
        changes = []
        
        # 펀딩비 데이터가 없는 경우 빈 리스트 반환
        if 'funding_rate' not in df.columns or df['funding_rate'].sum() == 0:
            return changes
        
        # 전체 데이터 순회하며 중요 변동점 찾기
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # 1. 펀딩비 부호 변경
            sign_change = current['funding_sign_change']
            
            # 2. 큰 펀딩비 값 (절대값 0.01% 이상)
            significant_value = abs(current['funding_rate']) > 0.0001
            
            # 3. 큰 펀딩비 변화
            significant_change = abs(current['funding_rate'] - previous['funding_rate']) > 0.0001
            
            if sign_change or significant_value or significant_change:
                # timestamp 처리 - numpy.int64 타입 처리
                timestamp = current.name
                if hasattr(timestamp, 'isoformat'):
                    timestamp = timestamp.isoformat()
                else:
                    timestamp = str(timestamp)
                    
                changes.append({
                    'timestamp': timestamp,
                    'price': float(current['close']),
                    'funding_rate': float(current['funding_rate']),
                    'funding_prev': float(previous['funding_rate']),
                    'type': 'sign_change' if sign_change else \
                           'significant_value' if significant_value else 'significant_change',
                    'direction': 'positive' if current['funding_rate'] > 0 else 'negative'
                })
        
        # 중요도 순으로 정렬 (변화율 크기)
        changes.sort(key=lambda x: abs(float(x['funding_rate'])), reverse=True)
        
        # 상위 10개만 반환
        return changes[:10]
        
    def _find_volume_spikes(self, df: pd.DataFrame) -> List[Dict]:
        """거래량 급증 지점 찾기"""
        spikes = []
        
        # 이동평균 계산
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 전체 데이터 순회하며 중요 거래량 급증 찾기
        for i in range(20, len(df)):
            current = df.iloc[i]
            
            # 거래량 급증 (이동평균의 3배 이상)
            if current['volume_ratio'] > 3.0:
                # timestamp 처리 - numpy.int64 타입 처리
                timestamp = current.name
                if hasattr(timestamp, 'isoformat'):
                    timestamp = timestamp.isoformat()
                else:
                    timestamp = str(timestamp)
                    
                spikes.append({
                    'timestamp': timestamp,
                    'price': float(current['close']),
                    'volume': float(current['volume']),
                    'volume_ratio': float(current['volume_ratio']),
                    'direction': 'up' if current['close'] > df.iloc[i-1]['close'] else 'down'
                })
        
        # 거래량 비율 순으로 정렬
        spikes.sort(key=lambda x: float(x['volume_ratio']), reverse=True)
        
        # 상위 10개만 반환
        return spikes[:10]
        
    def analyze(self) -> Dict:
        """심볼 분석 실행"""
        try:
            # 1. 데이터 로드
            if not self.load_data():
                return {}
                
            # 2. 멀티 타임프레임 데이터 준비
            self.prepare_timeframes()
            if not self.timeframe_data:
                return {}
                
            # 3. 중요 변동점 찾기
            results = self.find_significant_changes()
                
            # 4. 결과 저장
            self.save_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"{self.symbol} 분석 중 오류: {str(e)}")
            return {}
            
    def save_results(self, results: Dict):
        """분석 결과 저장"""
        try:
            # 결과 파일 저장
            output_dir = Path("data/results")
            output_dir.mkdir(parents=True, exist_ok=True)
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
            
        except Exception as e:
            logger.error(f"분석 실행 중 오류: {str(e)}")
            
def main():
    """메인 실행 함수"""
    worker = AnalysisWorker(max_workers=4)
    worker.run()

if __name__ == "__main__":
    main()