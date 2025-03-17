#!/usr/bin/env python
"""
기존 통합 JSON 파일을 각 심볼별 개별 파일로 변환하는 스크립트
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("convert_to_individual.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ConvertToIndividual")

def convert_to_individual_files(source_file, output_dir="data/live"):
    """통합 JSON 파일을 각 심볼별 개별 파일로 변환하고 통합 요약 파일 생성"""
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 요약 파일 디렉토리 생성
    summary_dir = Path("data/signals")
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"소스 파일 '{source_file}'에서 개별 파일 변환 시작")
    
    try:
        # 소스 파일 로드
        with open(source_file, 'r') as f:
            source_data = json.load(f)
        
        timestamp = source_data.get('timestamp', datetime.now().isoformat())
        best_signals = source_data.get('best_signals', {})
        total_symbols = len(best_signals)
        
        logger.info(f"총 {total_symbols}개 심볼 발견")
        
        # 통합 요약 데이터 초기화
        summary_data = {
            "timestamp": timestamp,
            "total_symbols": total_symbols,
            "symbols_data": {},
            "statistics": {
                "event_type_distribution": {},
                "timeframe_distribution": {}
            },
            "last_updated": datetime.now().isoformat()
        }
        
        # 각 심볼별로 개별 파일 생성
        created_files = 0
        for symbol, signal_data in best_signals.items():
            # 심볼별 데이터 구성
            symbol_data = {
                "symbol": symbol,
                "signals": [signal_data],  # 신호를 리스트로 포함
                "best_signal": signal_data,
                "analyzed_at": timestamp
            }
            
            # 파일 저장
            output_file = output_path / f"{symbol}.json"
            with open(output_file, 'w') as f:
                json.dump(symbol_data, f, indent=2)
            
            # 요약 데이터에 추가
            summary_data["symbols_data"][symbol] = {
                "symbol": symbol,
                "best_signal": signal_data,
                "total_signals": 1,
                "analyzed_at": timestamp
            }
            
            # 통계 정보 업데이트
            event_type = signal_data.get('event_type')
            timeframe = signal_data.get('timeframe')
            
            if event_type:
                summary_data["statistics"]["event_type_distribution"][event_type] = summary_data["statistics"]["event_type_distribution"].get(event_type, 0) + 1
            if timeframe:
                summary_data["statistics"]["timeframe_distribution"][timeframe] = summary_data["statistics"]["timeframe_distribution"].get(timeframe, 0) + 1
            
            created_files += 1
            
            if created_files % 50 == 0:
                logger.info(f"{created_files}/{total_symbols} 파일 생성 완료")
        
        # 통합 요약 파일 저장
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = summary_dir / f"all_symbols_summary_{timestamp_str}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"변환 완료: {created_files}개 파일이 '{output_dir}' 디렉토리에 생성됨")
        logger.info(f"통합 요약 파일 생성 완료: {summary_file}")
        
        return created_files
        
    except Exception as e:
        logger.error(f"변환 중 오류 발생: {e}")
        return 0

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="통합 JSON 파일을 각 심볼별 개별 파일로 변환")
    parser.add_argument("--source", type=str, default="data/alarms/all_symbols_best_signals_20250312_125127.json", 
                        help="소스 JSON 파일 경로")
    parser.add_argument("--output-dir", type=str, default="data/live", 
                        help="출력 디렉토리 경로")
    
    args = parser.parse_args()
    
    convert_to_individual_files(args.source, args.output_dir)

if __name__ == "__main__":
    main() 