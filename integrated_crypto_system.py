#!/usr/bin/env python
"""
통합 암호화폐 분석 시스템
- 패턴 감지 (SymbolAnalyzer)
- 병렬 처리 (AnalysisWorker)
- 승률 기반 신호 생성 (CryptoAlarmSystem)
"""

import argparse
import logging
import time
import os
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integrated_crypto_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IntegratedCryptoSystem")

def setup_directories():
    """필요한 디렉토리 생성"""
    dirs = [
        "data/live",
        "data/results",
        "data/summary",
        "data/alarms",
        "data/config"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 기본 설정 파일이 없으면 생성
    default_config = Path("data/config/default.json")
    if not default_config.exists():
        with open(default_config, 'w') as f:
            json.dump({
                "min_data_points": 20,
                "oi_change_threshold": 0.05,
                "funding_threshold": 0.001,
                "volume_ratio_threshold": 3.0
            }, f, indent=2)
        logger.info("기본 설정 파일 생성 완료")

def run_pattern_analysis(workers=4, symbol=None):
    """패턴 분석 실행"""
    try:
        cmd = [sys.executable, "crypto_pattern_analyzer_parallel.py"]
        
        if workers:
            cmd.extend(["--workers", str(workers)])
        
        if symbol:
            cmd.extend(["--symbol", symbol])
        
        logger.info(f"패턴 분석 명령 실행: {' '.join(cmd)}")
        
        # 서브프로세스로 실행
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode == 0:
            logger.info("패턴 분석 성공적으로 완료")
            if process.stdout:
                logger.info(f"출력: {process.stdout}")
            return True
        else:
            logger.error(f"패턴 분석 실패 (코드: {process.returncode})")
            if process.stderr:
                logger.error(f"오류: {process.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"패턴 분석 실행 중 오류 발생: {str(e)}")
        return False

def run_signal_analysis(min_score=70.0, symbol=None):
    """승률 기반 신호 분석 실행"""
    try:
        cmd = [sys.executable, "crypto_alarm_system.py"]
        
        cmd.extend(["--min-score", str(min_score)])
        
        if symbol:
            cmd.extend(["--symbol", symbol])
        else:
            cmd.append("--analyze-only")
        
        logger.info(f"신호 분석 명령 실행: {' '.join(cmd)}")
        
        # 서브프로세스로 실행
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode == 0:
            logger.info("신호 분석 성공적으로 완료")
            if process.stdout:
                logger.info(f"출력: {process.stdout}")
            return True
        else:
            logger.error(f"신호 분석 실패 (코드: {process.returncode})")
            if process.stderr:
                logger.error(f"오류: {process.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"신호 분석 실행 중 오류 발생: {str(e)}")
        return False

def generate_integrated_report():
    """통합 보고서 생성"""
    try:
        # 1. 패턴 분석 결과 로드
        pattern_summary_dir = Path("data/summary")
        pattern_summary_files = list(pattern_summary_dir.glob("analysis_summary_*.json"))
        
        if not pattern_summary_files:
            logger.warning("패턴 분석 요약 파일이 없습니다")
            pattern_summary = {}
        else:
            latest_pattern_summary = max(pattern_summary_files, key=lambda x: x.stat().st_mtime)
            with open(latest_pattern_summary, 'r') as f:
                pattern_summary = json.load(f)
        
        # 2. 신호 분석 결과 로드
        signal_file = Path("data/alarms/win_rate_analysis.json")
        if not signal_file.exists():
            logger.warning("신호 분석 결과 파일이 없습니다")
            signal_data = {}
        else:
            with open(signal_file, 'r') as f:
                signal_data = json.load(f)
        
        # 3. 통합 보고서 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path("data/alarms") / f"integrated_report_{timestamp}.json"
        
        # 상위 신호 추출
        top_signals = []
        for symbol, data in signal_data.get('symbols', {}).items():
            for signal in data.get('best_signals', []):
                top_signals.append({
                    'symbol': symbol,
                    'timeframe': signal.get('timeframe', ''),
                    'event_type': signal.get('event_type', ''),
                    'period': signal.get('period', ''),
                    'score': signal.get('score', 0)
                })
        
        # 점수 기준 내림차순 정렬
        top_signals.sort(key=lambda x: x['score'], reverse=True)
        
        # 패턴 정보 추출
        pattern_info = {}
        results_dir = Path("data/results")
        pattern_files = list(results_dir.glob("*_changes.json"))
        
        for pattern_file in pattern_files:
            try:
                with open(pattern_file, 'r') as f:
                    data = json.load(f)
                
                symbol = data.get('symbol', '')
                pattern_info[symbol] = {
                    'patterns': []
                }
                
                for tf, tf_data in data.get('results', {}).items():
                    for pattern in tf_data.get('patterns', []):
                        pattern_info[symbol]['patterns'].append({
                            'timeframe': tf,
                            'type': pattern.get('type', ''),
                            'confidence': pattern.get('confidence', 0)
                        })
            except Exception as e:
                logger.error(f"{pattern_file} 처리 중 오류: {str(e)}")
        
        # 통합 보고서 작성
        integrated_report = {
            'timestamp': datetime.now().isoformat(),
            'pattern_analysis': pattern_summary.get('stats', {}),
            'signal_analysis': {
                'total_signals': len(top_signals),
                'top_signals': top_signals[:20]  # 상위 20개
            },
            'combined_insights': []
        }
        
        # 패턴과 신호를 결합한 인사이트 생성
        for signal in top_signals[:20]:
            symbol = signal['symbol']
            if symbol in pattern_info and pattern_info[symbol]['patterns']:
                # 해당 심볼의 패턴 중 가장 높은 신뢰도를 가진 패턴 찾기
                best_pattern = max(pattern_info[symbol]['patterns'], 
                                  key=lambda x: x['confidence'])
                
                integrated_report['combined_insights'].append({
                    'symbol': symbol,
                    'signal': signal,
                    'pattern': best_pattern,
                    'combined_score': (signal['score'] + best_pattern['confidence']) / 2
                })
        
        # 통합 점수 기준 내림차순 정렬
        integrated_report['combined_insights'].sort(
            key=lambda x: x['combined_score'], reverse=True
        )
        
        # 보고서 저장
        with open(report_file, 'w') as f:
            json.dump(integrated_report, f, indent=2)
        
        logger.info(f"통합 보고서 생성 완료: {report_file}")
        
        # 상위 5개 인사이트 로그 출력
        if integrated_report['combined_insights']:
            logger.info("상위 5개 통합 인사이트:")
            for i, insight in enumerate(integrated_report['combined_insights'][:5], 1):
                logger.info(f"{i}. {insight['symbol']}: "
                           f"신호({insight['signal']['event_type']}, {insight['signal']['score']}점) + "
                           f"패턴({insight['pattern']['type']}, {insight['pattern']['confidence']}점) = "
                           f"통합 점수 {insight['combined_score']:.1f}점")
        
        return True
        
    except Exception as e:
        logger.error(f"통합 보고서 생성 중 오류 발생: {str(e)}")
        return False

def run_integrated_analysis(workers=4, min_score=70.0, symbol=None):
    """통합 분석 실행"""
    # 1. 패턴 분석 실행
    pattern_success = run_pattern_analysis(workers, symbol)
    
    if not pattern_success:
        logger.error("패턴 분석 실패로 인해 통합 분석이 중단되었습니다")
        return False
    
    # 2. 신호 분석 실행
    signal_success = run_signal_analysis(min_score, symbol)
    
    if not signal_success:
        logger.error("신호 분석 실패로 인해 통합 분석이 중단되었습니다")
        return False
    
    # 3. 통합 보고서 생성 (특정 심볼 분석이 아닌 경우에만)
    if not symbol:
        report_success = generate_integrated_report()
        if not report_success:
            logger.error("통합 보고서 생성 실패")
            return False
    
    logger.info("통합 분석이 성공적으로 완료되었습니다")
    return True

def run_periodic(interval_minutes=60, workers=4, min_score=70.0):
    """주기적 통합 분석 실행"""
    logger.info(f"주기적 통합 분석 시작 (간격: {interval_minutes}분, 작업자: {workers}명, 최소 점수: {min_score})")
    
    try:
        while True:
            start_time = time.time()
            
            # 통합 분석 실행
            success = run_integrated_analysis(workers, min_score)
            
            # 다음 실행 시간 계산
            elapsed = time.time() - start_time
            sleep_time = max(0, interval_minutes * 60 - elapsed)
            
            logger.info(f"다음 분석까지 {sleep_time:.1f}초 대기 중...")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중지됨")
    except Exception as e:
        logger.error(f"주기적 실행 중 오류 발생: {str(e)}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='통합 암호화폐 분석 시스템')
    parser.add_argument('--interval', type=int, default=60, help='주기적 실행 간격 (분)')
    parser.add_argument('--workers', type=int, default=4, help='병렬 작업자 수')
    parser.add_argument('--min-score', type=float, default=70.0, help='최소 신호 점수')
    parser.add_argument('--symbol', type=str, help='특정 심볼만 분석')
    parser.add_argument('--once', action='store_true', help='한 번만 실행')
    parser.add_argument('--pattern-only', action='store_true', help='패턴 분석만 실행')
    parser.add_argument('--signal-only', action='store_true', help='신호 분석만 실행')
    parser.add_argument('--report-only', action='store_true', help='통합 보고서만 생성')
    
    args = parser.parse_args()
    
    # 디렉토리 설정
    setup_directories()
    
    if args.pattern_only:
        # 패턴 분석만 실행
        run_pattern_analysis(args.workers, args.symbol)
    elif args.signal_only:
        # 신호 분석만 실행
        run_signal_analysis(args.min_score, args.symbol)
    elif args.report_only:
        # 통합 보고서만 생성
        generate_integrated_report()
    elif args.once or args.symbol:
        # 한 번만 통합 분석 실행
        run_integrated_analysis(args.workers, args.min_score, args.symbol)
    else:
        # 주기적 통합 분석 실행
        run_periodic(args.interval, args.workers, args.min_score)

if __name__ == "__main__":
    main() 