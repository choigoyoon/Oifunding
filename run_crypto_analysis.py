#!/usr/bin/env python
"""
암호화폐 패턴 분석 시스템 실행 스크립트
- 단일 실행 또는 주기적 실행 지원
- 결과 요약 및 알림 기능
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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_analysis_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoAnalysisRunner")

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

def run_analysis(workers=4, symbol=None):
    """분석 실행"""
    try:
        cmd = [sys.executable, "crypto_pattern_analyzer_parallel.py"]
        
        if workers:
            cmd.extend(["--workers", str(workers)])
        
        if symbol:
            cmd.extend(["--symbol", symbol])
        
        logger.info(f"분석 명령 실행: {' '.join(cmd)}")
        
        # 서브프로세스로 실행
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.returncode == 0:
            logger.info("분석 성공적으로 완료")
            if process.stdout:
                logger.info(f"출력: {process.stdout}")
            return True
        else:
            logger.error(f"분석 실패 (코드: {process.returncode})")
            if process.stderr:
                logger.error(f"오류: {process.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"분석 실행 중 오류 발생: {str(e)}")
        return False

def generate_alerts():
    """알림 생성"""
    try:
        # 가장 최신 요약 파일 찾기
        summary_dir = Path("data/summary")
        summary_files = list(summary_dir.glob("analysis_summary_*.json"))
        
        if not summary_files:
            logger.warning("요약 파일이 없습니다")
            return
            
        latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
        
        # 요약 파일 로드
        with open(latest_summary, 'r') as f:
            summary = json.load(f)
        
        # 결과 디렉토리에서 모든 심볼 결과 로드
        results_dir = Path("data/results")
        pattern_files = list(results_dir.glob("*_changes.json"))
        
        # 중요 패턴 추출
        important_patterns = []
        
        for pattern_file in pattern_files:
            try:
                with open(pattern_file, 'r') as f:
                    data = json.load(f)
                
                symbol = data.get('symbol', '')
                
                for tf, tf_data in data.get('results', {}).items():
                    for pattern in tf_data.get('patterns', []):
                        # 신뢰도 80 이상인 패턴만 추출
                        if pattern.get('confidence', 0) >= 80:
                            important_patterns.append({
                                'symbol': symbol,
                                'timeframe': tf,
                                'pattern_type': pattern.get('type', ''),
                                'confidence': pattern.get('confidence', 0),
                                'timestamp': tf_data.get('timestamp', ''),
                                'metrics': pattern.get('metrics', {})
                            })
            except Exception as e:
                logger.error(f"{pattern_file} 처리 중 오류: {str(e)}")
        
        # 신뢰도 기준 내림차순 정렬
        important_patterns.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 알림 파일 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_file = Path("data/alarms") / f"alerts_{timestamp}.json"
        
        with open(alert_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': summary.get('stats', {}),
                'important_patterns': important_patterns[:20]  # 상위 20개만
            }, f, indent=2)
        
        logger.info(f"알림 파일 생성 완료: {alert_file}")
        logger.info(f"중요 패턴 {len(important_patterns)}개 중 상위 20개 추출")
        
        # 상위 5개 패턴 로그 출력
        for i, pattern in enumerate(important_patterns[:5], 1):
            logger.info(f"{i}. {pattern['symbol']} ({pattern['timeframe']}): "
                       f"{pattern['pattern_type']} - 신뢰도 {pattern['confidence']}%")
        
        return True
        
    except Exception as e:
        logger.error(f"알림 생성 중 오류 발생: {str(e)}")
        return False

def run_periodic(interval_minutes=60, workers=4):
    """주기적 실행"""
    logger.info(f"주기적 분석 시작 (간격: {interval_minutes}분, 작업자: {workers}명)")
    
    try:
        while True:
            start_time = time.time()
            
            # 분석 실행
            success = run_analysis(workers)
            
            if success:
                # 알림 생성
                generate_alerts()
            
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
    parser = argparse.ArgumentParser(description='암호화폐 패턴 분석 시스템 실행')
    parser.add_argument('--interval', type=int, default=60, help='주기적 실행 간격 (분)')
    parser.add_argument('--workers', type=int, default=4, help='병렬 작업자 수')
    parser.add_argument('--symbol', type=str, help='특정 심볼만 분석')
    parser.add_argument('--once', action='store_true', help='한 번만 실행')
    parser.add_argument('--alerts-only', action='store_true', help='알림만 생성')
    
    args = parser.parse_args()
    
    # 디렉토리 설정
    setup_directories()
    
    if args.alerts_only:
        # 알림만 생성
        generate_alerts()
    elif args.once or args.symbol:
        # 한 번만 실행
        success = run_analysis(args.workers, args.symbol)
        if success and not args.symbol:
            generate_alerts()
    else:
        # 주기적 실행
        run_periodic(args.interval, args.workers)

if __name__ == "__main__":
    main() 