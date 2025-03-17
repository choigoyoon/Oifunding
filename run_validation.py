import os
import argparse
from symbol_signal_validator import SymbolSignalValidator

def main():
    """
    수정된 CSV 파일을 사용하여 검증 시스템을 실행하는 메인 함수
    """
    parser = argparse.ArgumentParser(description='심볼별 신호 검증 및 점수화 시스템 실행')
    parser.add_argument('--data-dir', type=str, default='data/formatted', help='수정된 CSV 데이터 디렉토리')
    parser.add_argument('--results-dir', type=str, default='data/results', help='JSON 결과 디렉토리')
    parser.add_argument('--output-dir', type=str, default='data/validated_with_timestamp', help='검증 결과 출력 디렉토리')
    parser.add_argument('--min-score', type=float, default=30.0, help='최소 점수')
    parser.add_argument('--lookback-days', type=int, default=30, help='검증을 위한 과거 데이터 조회 일수')
    
    args = parser.parse_args()
    
    # 검증 시스템 초기화 및 실행
    validator = SymbolSignalValidator(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        min_score=args.min_score,
        validation_lookback_days=args.lookback_days
    )
    
    validator.run()

if __name__ == "__main__":
    main() 