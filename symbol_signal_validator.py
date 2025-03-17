import os
import json
import glob
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# int64 타입을 JSON으로 직렬화하기 위한 클래스
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class SymbolSignalValidator:
    """
    각 심볼별로 JSON 파일을 생성하고 CSV로 검증한 후 점수화하는 클래스
    """
    
    def __init__(self, 
                 data_dir: str = 'data/live', 
                 results_dir: str = 'data/results',
                 output_dir: str = 'data/validated',
                 log_dir: str = 'logs',
                 min_score: float = 30.0,
                 validation_lookback_days: int = 30):
        """
        초기화
        
        Args:
            data_dir: CSV 데이터 디렉토리
            results_dir: JSON 결과 디렉토리
            output_dir: 검증 결과 출력 디렉토리
            log_dir: 로그 파일 디렉토리
            min_score: 최소 점수 (기본값: 30.0)
            validation_lookback_days: 검증을 위한 과거 데이터 조회 일수 (기본값: 30)
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.min_score = min_score
        self.validation_lookback_days = validation_lookback_days
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'symbols'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'csv_validation'), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 로깅 설정
        self._setup_logging()
        
        # 결과 저장 변수
        self.all_symbols = []
        self.symbol_signals = {}
        self.validation_results = {}
        self.final_scores = {}
        
        self.logger.info("SymbolSignalValidator 초기화 완료")
        self.logger.info(f"데이터 디렉토리: {data_dir}")
        self.logger.info(f"결과 디렉토리: {results_dir}")
        self.logger.info(f"출력 디렉토리: {output_dir}")
        self.logger.info(f"최소 점수: {min_score}")
        self.logger.info(f"검증 조회 일수: {validation_lookback_days}")
    
    def _setup_logging(self):
        """
        로깅 설정
        """
        # 로거 생성
        self.logger = logging.getLogger('symbol_signal_validator')
        self.logger.setLevel(logging.DEBUG)
        
        # 이미 핸들러가 있으면 제거
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 현재 시간을 파일명에 포함
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f'validation_{timestamp}.log')
        
        # 파일 핸들러 추가
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러 추가
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 등록
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"로깅 설정 완료. 로그 파일: {log_file}")
    
    def run(self):
        """
        검증 시스템 실행
        """
        self.logger.info("=" * 50)
        self.logger.info("심볼별 신호 검증 및 점수화 시스템 실행")
        self.logger.info("=" * 50)
        
        print("=" * 50)
        print("심볼별 신호 검증 및 점수화 시스템 실행")
        print("=" * 50)
        
        # 1. 모든 심볼 목록 가져오기
        self._get_all_symbols()
        
        # 2. 각 심볼별 JSON 파일 생성
        self._generate_symbol_json_files()
        
        # 3. CSV 데이터로 검증
        self._validate_with_csv()
        
        # 4. 최종 점수화 및 결과 저장
        self._calculate_final_scores()
        
        # 5. 요약 보고서 생성
        self._generate_summary_report()
        
        self.logger.info("심볼별 신호 검증 및 점수화 시스템 실행 완료")
    
    def _get_all_symbols(self):
        """
        모든 심볼 목록 가져오기
        """
        self.logger.info("1. 모든 심볼 목록 가져오기...")
        print("\n1. 모든 심볼 목록 가져오기...")
        
        # JSON 결과 파일에서 심볼 목록 가져오기
        json_files = glob.glob(os.path.join(self.results_dir, '*_changes.json'))
        self.all_symbols = [os.path.basename(f).replace('_changes.json', '') for f in json_files]
        
        self.logger.info(f"총 {len(self.all_symbols)}개 심볼을 찾았습니다.")
        print(f"총 {len(self.all_symbols)}개 심볼을 찾았습니다.")
    
    def _generate_symbol_json_files(self):
        """
        각 심볼별 JSON 파일 생성
        """
        self.logger.info("2. 각 심볼별 JSON 파일 생성 중...")
        print("\n2. 각 심볼별 JSON 파일 생성 중...")
        
        for symbol in self.all_symbols:
            self._process_symbol(symbol)
        
        self.logger.info(f"총 {len(self.symbol_signals)}개 심볼의 JSON 파일을 생성했습니다.")
        print(f"총 {len(self.symbol_signals)}개 심볼의 JSON 파일을 생성했습니다.")
    
    def _process_symbol(self, symbol: str):
        """
        단일 심볼 처리
        
        Args:
            symbol: 심볼명
        """
        # 심볼의 JSON 결과 파일 경로
        json_file = os.path.join(self.results_dir, f"{symbol}_changes.json")
        
        if not os.path.exists(json_file):
            self.logger.warning(f"{symbol}: JSON 파일을 찾을 수 없습니다.")
            print(f"  - {symbol}: JSON 파일을 찾을 수 없습니다.")
            return
        
        try:
            # JSON 파일 로드
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 심볼 데이터 추출
            symbol_data = {
                'symbol': symbol,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'timeframes': {}
            }
            
            # 각 타임프레임별 데이터 추출 - 수정된 부분
            if 'results' in data:
                for tf in ['1M', '5M', '15M', '30M', '1H']:
                    if tf in data['results']:
                        tf_data = {
                            'oi_changes': data['results'][tf].get('oi_changes', []),
                            'funding_changes': data['results'][tf].get('funding_changes', []),
                            'volume_spikes': data['results'][tf].get('volume_spikes', [])
                        }
                        
                        # 이벤트 수 계산
                        event_counts = {
                            'oi_changes': len(tf_data['oi_changes']),
                            'funding_changes': len(tf_data['funding_changes']),
                            'volume_spikes': len(tf_data['volume_spikes'])
                        }
                        
                        # 초기 점수 계산 (이벤트 수 기반)
                        initial_scores = {
                            'oi_changes': min(100, event_counts['oi_changes'] * 10),
                            'funding_changes': min(100, event_counts['funding_changes'] * 10),
                            'volume_spikes': min(100, event_counts['volume_spikes'] * 5)
                        }
                        
                        tf_data['event_counts'] = event_counts
                        tf_data['initial_scores'] = initial_scores
                        
                        symbol_data['timeframes'][tf] = tf_data
            
            # 심볼 데이터 저장
            self.symbol_signals[symbol] = symbol_data
            
            # 심볼별 JSON 파일 저장
            output_file = os.path.join(self.output_dir, 'symbols', f"{symbol}_signals.json")
            with open(output_file, 'w') as f:
                json.dump(symbol_data, f, indent=2, cls=NumpyEncoder)
            
            self.logger.debug(f"{symbol}: JSON 파일 생성 완료")
            print(f"  - {symbol}: JSON 파일 생성 완료")
            
        except Exception as e:
            self.logger.error(f"{symbol}: 처리 중 오류 발생 - {str(e)}")
            print(f"  - {symbol}: 처리 중 오류 발생 - {str(e)}")
    
    def _validate_with_csv(self):
        """
        CSV 데이터로 검증
        """
        self.logger.info("3. CSV 데이터로 검증 중...")
        print("\n3. CSV 데이터로 검증 중...")
        
        for symbol in self.symbol_signals:
            self._validate_symbol(symbol)
        
        self.logger.info(f"총 {len(self.validation_results)}개 심볼의 검증을 완료했습니다.")
        print(f"총 {len(self.validation_results)}개 심볼의 검증을 완료했습니다.")
    
    def _validate_symbol(self, symbol: str):
        """
        단일 심볼 검증
        
        Args:
            symbol: 심볼명
        """
        # 심볼의 CSV 파일 경로
        csv_file = os.path.join(self.data_dir, f"{symbol}.csv")
        
        if not os.path.exists(csv_file):
            self.logger.warning(f"{symbol}: CSV 파일을 찾을 수 없습니다.")
            print(f"  - {symbol}: CSV 파일을 찾을 수 없습니다.")
            self.validation_results[symbol] = {
                'status': 'failed',
                'reason': 'CSV file not found',
                'validation_scores': {
                    'oi_changes': 0,
                    'funding_changes': 0,
                    'volume_spikes': 0
                }
            }
            return
        
        try:
            # CSV 파일 로드
            df = pd.read_csv(csv_file)
            
            # datetime 열이 있는지 확인
            if 'datetime' not in df.columns:
                self.logger.warning(f"{symbol}: CSV 파일에 datetime 열이 없습니다.")
                print(f"  - {symbol}: CSV 파일에 datetime 열이 없습니다.")
                self.validation_results[symbol] = {
                    'status': 'failed',
                    'reason': 'No datetime column in CSV',
                    'validation_scores': {
                        'oi_changes': 0,
                        'funding_changes': 0,
                        'volume_spikes': 0
                    }
                }
                return
            
            # datetime을 timestamp로 변환
            df['timestamp'] = pd.to_datetime(df['datetime'])
            
            # 최근 N일 데이터만 사용
            cutoff_date = datetime.now() - pd.Timedelta(days=self.validation_lookback_days)
            recent_df = df[df['timestamp'] >= cutoff_date]
            
            if len(recent_df) == 0:
                self.logger.warning(f"{symbol}: 최근 {self.validation_lookback_days}일 데이터가 없습니다.")
                print(f"  - {symbol}: 최근 {self.validation_lookback_days}일 데이터가 없습니다.")
                self.validation_results[symbol] = {
                    'status': 'failed',
                    'reason': f'No data in the last {self.validation_lookback_days} days',
                    'validation_scores': {
                        'oi_changes': 0,
                        'funding_changes': 0,
                        'volume_spikes': 0
                    }
                }
                return
            
            # 검증 점수 계산
            validation_scores = {
                'oi_changes': 0,
                'funding_changes': 0,
                'volume_spikes': 0
            }
            
            # 오픈 인터레스트 검증
            if 'open_interest' in recent_df.columns:
                # NaN 값 제거
                oi_df = recent_df.dropna(subset=['open_interest'])
                if len(oi_df) > 0:
                    oi_changes = oi_df['open_interest'].pct_change().abs()
                    significant_oi_changes = oi_changes[oi_changes > 0.05].count()
                    validation_scores['oi_changes'] = min(100, int(significant_oi_changes * 5))
                    self.logger.debug(f"{symbol}: 오픈 인터레스트 검증 점수 - {validation_scores['oi_changes']}")
            
            # 펀딩 레이트 검증
            if 'funding_rate' in recent_df.columns:
                # NaN 값 제거
                funding_df = recent_df.dropna(subset=['funding_rate'])
                if len(funding_df) > 0:
                    funding_changes = funding_df['funding_rate'].diff().abs()
                    significant_funding_changes = funding_changes[funding_changes > 0.0001].count()
                    validation_scores['funding_changes'] = min(100, int(significant_funding_changes * 10))
                    self.logger.debug(f"{symbol}: 펀딩 레이트 검증 점수 - {validation_scores['funding_changes']}")
            
            # 볼륨 스파이크 검증
            if 'volume' in recent_df.columns:
                # NaN 값 제거
                volume_df = recent_df.dropna(subset=['volume'])
                if len(volume_df) > 0:
                    volume_mean = volume_df['volume'].mean()
                    volume_spikes = volume_df[volume_df['volume'] > volume_mean * 2].count()['volume']
                    validation_scores['volume_spikes'] = min(100, int(volume_spikes * 5))
                    self.logger.debug(f"{symbol}: 볼륨 스파이크 검증 점수 - {validation_scores['volume_spikes']}")
            
            # 검증 결과 저장
            self.validation_results[symbol] = {
                'status': 'success',
                'data_points': int(len(recent_df)),
                'validation_scores': validation_scores
            }
            
            # CSV 검증 결과 저장
            output_file = os.path.join(self.output_dir, 'csv_validation', f"{symbol}_validation.json")
            with open(output_file, 'w') as f:
                json.dump(self.validation_results[symbol], f, indent=2, cls=NumpyEncoder)
            
            self.logger.debug(f"{symbol}: CSV 검증 완료")
            print(f"  - {symbol}: CSV 검증 완료")
            
        except Exception as e:
            self.logger.error(f"{symbol}: 검증 중 오류 발생 - {str(e)}")
            print(f"  - {symbol}: 검증 중 오류 발생 - {str(e)}")
            self.validation_results[symbol] = {
                'status': 'failed',
                'reason': str(e),
                'validation_scores': {
                    'oi_changes': 0,
                    'funding_changes': 0,
                    'volume_spikes': 0
                }
            }
    
    def _calculate_final_scores(self):
        """
        최종 점수화 및 결과 저장
        """
        self.logger.info("4. 최종 점수화 및 결과 저장 중...")
        print("\n4. 최종 점수화 및 결과 저장 중...")
        
        for symbol in self.symbol_signals:
            if symbol in self.validation_results:
                self._calculate_symbol_final_score(symbol)
        
        # 최종 점수 기준으로 정렬
        sorted_symbols = sorted(self.final_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        
        # 최종 결과 저장
        final_results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_symbols': len(sorted_symbols),
            'min_score': self.min_score,
            'symbols': dict(sorted_symbols)
        }
        
        output_file = os.path.join(self.output_dir, 'final_scores.json')
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"최종 점수화 완료. 결과가 {output_file}에 저장되었습니다.")
        print(f"최종 점수화 완료. 결과가 {output_file}에 저장되었습니다.")
        
        # 상위 10개 심볼 로깅
        top_symbols = sorted_symbols[:10]
        self.logger.info("상위 10개 심볼:")
        for symbol, data in top_symbols:
            self.logger.info(f"  - {symbol}: {data['total_score']:.2f} 점")
    
    def _calculate_symbol_final_score(self, symbol: str):
        """
        단일 심볼의 최종 점수 계산
        
        Args:
            symbol: 심볼명
        """
        symbol_data = self.symbol_signals[symbol]
        validation_data = self.validation_results[symbol]
        
        # 타임프레임별 점수 계산
        timeframe_scores = {}
        for tf, tf_data in symbol_data['timeframes'].items():
            # 이벤트 유형별 점수 계산
            event_scores = {}
            for event_type in ['oi_changes', 'funding_changes', 'volume_spikes']:
                # JSON 초기 점수와 CSV 검증 점수의 가중 평균
                initial_score = tf_data['initial_scores'].get(event_type, 0)
                validation_score = validation_data['validation_scores'].get(event_type, 0)
                
                # 가중치: JSON 60%, CSV 40%
                weighted_score = (initial_score * 0.6) + (validation_score * 0.4)
                event_scores[event_type] = weighted_score
            
            # 타임프레임 평균 점수
            avg_score = sum(event_scores.values()) / len(event_scores) if event_scores else 0
            timeframe_scores[tf] = {
                'event_scores': event_scores,
                'avg_score': avg_score
            }
        
        # 전체 평균 점수
        total_score = sum(tf_data['avg_score'] for tf_data in timeframe_scores.values()) / len(timeframe_scores) if timeframe_scores else 0
        
        # 최종 점수 저장
        self.final_scores[symbol] = {
            'timeframe_scores': timeframe_scores,
            'total_score': total_score,
            'validation_status': validation_data['status'],
            'pass': total_score >= self.min_score
        }
        
        self.logger.debug(f"{symbol}: 최종 점수 - {total_score:.2f}, 통과 여부 - {total_score >= self.min_score}")
    
    def _generate_summary_report(self):
        """
        요약 보고서 생성
        """
        self.logger.info("5. 요약 보고서 생성 중...")
        print("\n5. 요약 보고서 생성 중...")
        
        # 통계 계산
        total_symbols = len(self.final_scores)
        passed_symbols = sum(1 for data in self.final_scores.values() if data['pass'])
        failed_symbols = total_symbols - passed_symbols
        
        # 이벤트 유형별 평균 점수
        event_scores = defaultdict(list)
        for symbol, data in self.final_scores.items():
            for tf, tf_data in data['timeframe_scores'].items():
                for event_type, score in tf_data['event_scores'].items():
                    event_scores[event_type].append(score)
        
        avg_event_scores = {event_type: sum(scores) / len(scores) if scores else 0 
                           for event_type, scores in event_scores.items()}
        
        # 타임프레임별 평균 점수
        tf_scores = defaultdict(list)
        for symbol, data in self.final_scores.items():
            for tf, tf_data in data['timeframe_scores'].items():
                tf_scores[tf].append(tf_data['avg_score'])
        
        avg_tf_scores = {tf: sum(scores) / len(scores) if scores else 0 
                        for tf, scores in tf_scores.items()}
        
        # 상위 10개 심볼
        top_symbols = sorted(self.final_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)[:10]
        
        # 요약 보고서 생성
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_symbols': total_symbols,
            'passed_symbols': passed_symbols,
            'failed_symbols': failed_symbols,
            'pass_rate': passed_symbols / total_symbols if total_symbols > 0 else 0,
            'min_score': self.min_score,
            'avg_event_scores': avg_event_scores,
            'avg_timeframe_scores': avg_tf_scores,
            'top_symbols': [{
                'symbol': symbol,
                'total_score': data['total_score'],
                'validation_status': data['validation_status']
            } for symbol, data in top_symbols]
        }
        
        # 요약 보고서 저장
        output_file = os.path.join(self.output_dir, 'summary_report.json')
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        # CSV 요약 보고서 생성
        self._generate_csv_summary()
        
        self.logger.info(f"요약 보고서가 {output_file}에 저장되었습니다.")
        print(f"요약 보고서가 {output_file}에 저장되었습니다.")
        
        # 요약 통계 로깅
        self.logger.info(f"총 심볼 수: {total_symbols}")
        self.logger.info(f"통과 심볼 수: {passed_symbols} ({passed_symbols/total_symbols*100:.2f}%)")
        self.logger.info(f"실패 심볼 수: {failed_symbols} ({failed_symbols/total_symbols*100:.2f}%)")
        self.logger.info(f"이벤트 유형별 평균 점수: {avg_event_scores}")
        self.logger.info(f"타임프레임별 평균 점수: {avg_tf_scores}")
    
    def _generate_csv_summary(self):
        """
        CSV 요약 보고서 생성
        """
        # 심볼별 데이터 준비
        rows = []
        for symbol, data in self.final_scores.items():
            row = {
                'symbol': symbol,
                'total_score': data['total_score'],
                'pass': data['pass'],
                'validation_status': data['validation_status']
            }
            
            # 타임프레임별 점수 추가
            for tf, tf_data in data['timeframe_scores'].items():
                row[f'{tf}_score'] = tf_data['avg_score']
                
                # 이벤트 유형별 점수 추가
                for event_type, score in tf_data['event_scores'].items():
                    row[f'{tf}_{event_type}_score'] = score
            
            rows.append(row)
        
        # DataFrame 생성
        df = pd.DataFrame(rows)
        
        # CSV 파일로 저장
        output_file = os.path.join(self.output_dir, 'symbol_scores.csv')
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"CSV 요약 보고서가 {output_file}에 저장되었습니다.")
        print(f"CSV 요약 보고서가 {output_file}에 저장되었습니다.")

# 메인 실행 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='심볼별 신호 검증 및 점수화 시스템')
    parser.add_argument('--data-dir', type=str, default='data/live', help='CSV 데이터 디렉토리')
    parser.add_argument('--results-dir', type=str, default='data/results', help='JSON 결과 디렉토리')
    parser.add_argument('--output-dir', type=str, default='data/validated', help='검증 결과 출력 디렉토리')
    parser.add_argument('--log-dir', type=str, default='logs', help='로그 파일 디렉토리')
    parser.add_argument('--min-score', type=float, default=30.0, help='최소 점수')
    parser.add_argument('--lookback-days', type=int, default=30, help='검증을 위한 과거 데이터 조회 일수')
    
    args = parser.parse_args()
    
    # 검증 시스템 초기화 및 실행
    validator = SymbolSignalValidator(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        min_score=args.min_score,
        validation_lookback_days=args.lookback_days
    )
    
    validator.run() 