"""
데이터 저장 및 관리 모듈
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from ..utils.logger import logger
from ..config import DATA_DIR, LIVE_DIR, BACKUP_DIR, ARCHIVE_DIR, LONG_TERM_DIR

class DataManager:
    def __init__(self):
        """데이터 관리자 초기화"""
        self.data_dir = Path(DATA_DIR)
        self.live_dir = Path(LIVE_DIR)
        self.backup_dir = Path(BACKUP_DIR)
        self.archive_dir = Path(ARCHIVE_DIR)
        self.long_term_dir = Path(LONG_TERM_DIR)
        
        # 디렉토리 생성
        for directory in [self.data_dir, self.live_dir, self.backup_dir, 
                         self.archive_dir, self.long_term_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_data(self, df: pd.DataFrame, symbol: str, data_type: str) -> bool:
        """데이터 저장 (소수점 보존)"""
        try:
            if df.empty:
                logger.warning(f"{symbol} {data_type} 데이터가 비어있어 저장하지 않습니다.")
                return False
            
            # 파일 경로 설정
            file_path = self.data_dir / f"{symbol}_{data_type}.csv"
            
            # 기존 데이터 로드
            if file_path.exists():
                existing_df = pd.read_csv(file_path)
                # datetime 컬럼 변환
                existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                # 중복 제거 및 병합
                df = pd.concat([existing_df, df]).drop_duplicates(subset=['datetime'])
                df = df.sort_values('datetime')
            
            # 데이터 저장 (소수점 보존)
            df.to_csv(file_path, index=False, float_format='%.8f')
            logger.info(f"{symbol} {data_type} 데이터 저장 완료 ({len(df)}행)")
            return True
            
        except Exception as e:
            logger.error(f"{symbol} {data_type} 데이터 저장 중 오류: {str(e)}")
            return False
    
    def load_data(self, symbol: str, data_type: str) -> pd.DataFrame:
        """데이터 로드 (소수점 보존)"""
        try:
            file_path = self.data_dir / f"{symbol}_{data_type}.csv"
            if not file_path.exists():
                logger.warning(f"{symbol} {data_type} 데이터 파일이 없습니다.")
                return pd.DataFrame()
            
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            logger.info(f"{symbol} {data_type} 데이터 로드 완료 ({len(df)}행)")
            return df
            
        except Exception as e:
            logger.error(f"{symbol} {data_type} 데이터 로드 중 오류: {str(e)}")
            return pd.DataFrame()
    
    def backup_data(self, symbol: str, data_type: str) -> bool:
        """데이터 백업"""
        try:
            source_path = self.data_dir / f"{symbol}_{data_type}.csv"
            if not source_path.exists():
                logger.warning(f"{symbol} {data_type} 데이터 파일이 없어 백업하지 않습니다.")
                return False
            
            # 백업 파일명에 타임스탬프 추가
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{symbol}_{data_type}_{timestamp}.csv"
            
            # 파일 복사
            import shutil
            shutil.copy2(source_path, backup_path)
            logger.info(f"{symbol} {data_type} 데이터 백업 완료: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"{symbol} {data_type} 데이터 백업 중 오류: {str(e)}")
            return False
    
    def archive_data(self, symbol: str, data_type: str, days: int = 30) -> bool:
        """오래된 데이터 아카이브"""
        try:
            file_path = self.data_dir / f"{symbol}_{data_type}.csv"
            if not file_path.exists():
                logger.warning(f"{symbol} {data_type} 데이터 파일이 없어 아카이브하지 않습니다.")
                return False
            
            # 데이터 로드
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # 기준 날짜 설정
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 아카이브할 데이터와 유지할 데이터 분리
            archive_df = df[df['datetime'] < cutoff_date]
            keep_df = df[df['datetime'] >= cutoff_date]
            
            if archive_df.empty:
                logger.info(f"{symbol} {data_type} 아카이브할 데이터가 없습니다.")
                return True
            
            # 아카이브 파일 저장
            archive_path = self.archive_dir / f"{symbol}_{data_type}_archive_{datetime.now().strftime('%Y%m%d')}.csv"
            archive_df.to_csv(archive_path, index=False, float_format='%.8f')
            
            # 유지할 데이터만 다시 저장
            keep_df.to_csv(file_path, index=False, float_format='%.8f')
            
            logger.info(f"{symbol} {data_type} 데이터 아카이브 완료: {len(archive_df)}행 아카이브, {len(keep_df)}행 유지")
            return True
            
        except Exception as e:
            logger.error(f"{symbol} {data_type} 데이터 아카이브 중 오류: {str(e)}")
            return False
    
    def save_long_term_data(self, df: pd.DataFrame, symbol: str, data_type: str) -> bool:
        """장기 데이터 저장 (소수점 보존)"""
        try:
            if df.empty:
                logger.warning(f"{symbol} {data_type} 장기 데이터가 비어있어 저장하지 않습니다.")
                return False
            
            # 파일 경로 설정
            file_path = self.long_term_dir / f"{symbol}_{data_type}_long_term.csv"
            
            # 기존 데이터 로드
            if file_path.exists():
                existing_df = pd.read_csv(file_path)
                # datetime 컬럼 변환
                existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                # 중복 제거 및 병합
                df = pd.concat([existing_df, df]).drop_duplicates(subset=['datetime'])
                df = df.sort_values('datetime')
            
            # 데이터 저장 (소수점 보존)
            df.to_csv(file_path, index=False, float_format='%.8f')
            logger.info(f"{symbol} {data_type} 장기 데이터 저장 완료 ({len(df)}행)")
            return True
            
        except Exception as e:
            logger.error(f"{symbol} {data_type} 장기 데이터 저장 중 오류: {str(e)}")
            return False
    
    def load_long_term_data(self, symbol: str, data_type: str) -> pd.DataFrame:
        """장기 데이터 로드 (소수점 보존)"""
        try:
            file_path = self.long_term_dir / f"{symbol}_{data_type}_long_term.csv"
            if not file_path.exists():
                logger.warning(f"{symbol} {data_type} 장기 데이터 파일이 없습니다.")
                return pd.DataFrame()
            
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            logger.info(f"{symbol} {data_type} 장기 데이터 로드 완료 ({len(df)}행)")
            return df
            
        except Exception as e:
            logger.error(f"{symbol} {data_type} 장기 데이터 로드 중 오류: {str(e)}")
            return pd.DataFrame()
    
    def get_symbols(self) -> List[str]:
        """저장된 모든 심볼 목록 반환"""
        try:
            symbols = set()
            for file in self.data_dir.glob("*.csv"):
                symbol = file.stem.split('_')[0]
                symbols.add(symbol)
            return sorted(list(symbols))
        except Exception as e:
            logger.error(f"심볼 목록 로드 중 오류: {str(e)}")
            return []
    
    def get_data_types(self, symbol: str) -> List[str]:
        """특정 심볼의 데이터 타입 목록 반환"""
        try:
            data_types = set()
            for file in self.data_dir.glob(f"{symbol}_*.csv"):
                data_type = file.stem.split('_', 1)[1]
                data_types.add(data_type)
            return sorted(list(data_types))
        except Exception as e:
            logger.error(f"{symbol} 데이터 타입 목록 로드 중 오류: {str(e)}")
            return [] 