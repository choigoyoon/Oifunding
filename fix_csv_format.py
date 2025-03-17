import os
import glob
import pandas as pd
from datetime import datetime, timedelta

class CSVFormatter:
    """
    CSV 파일의 형식을 수정하여 timestamp 열을 추가하는 클래스
    """
    
    def __init__(self, 
                 data_dir: str = 'data/live',
                 output_dir: str = 'data/formatted',
                 days_back: int = 30):
        """
        초기화
        
        Args:
            data_dir: 원본 CSV 데이터 디렉토리
            output_dir: 형식이 수정된 CSV 출력 디렉토리
            days_back: 생성할 과거 데이터 일수 (기본값: 30)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.days_back = days_back
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self):
        """
        CSV 형식 수정 실행
        """
        print("=" * 50)
        print("CSV 파일 형식 수정 시스템 실행")
        print("=" * 50)
        
        # 모든 CSV 파일 목록 가져오기
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        print(f"\n총 {len(csv_files)}개의 CSV 파일을 찾았습니다.")
        
        # 각 CSV 파일 처리
        for csv_file in csv_files:
            self._process_csv_file(csv_file)
    
    def _process_csv_file(self, csv_file: str):
        """
        단일 CSV 파일 처리
        
        Args:
            csv_file: CSV 파일 경로
        """
        symbol = os.path.basename(csv_file).replace('.csv', '')
        print(f"  - {symbol} 처리 중...")
        
        try:
            # CSV 파일 로드
            df = pd.read_csv(csv_file)
            
            # 이미 timestamp 열이 있는지 확인
            if 'timestamp' in df.columns:
                print(f"    이미 timestamp 열이 있습니다. 건너뜁니다.")
                return
            
            # 현재 시간
            now = datetime.now()
            
            # 타임스탬프 열 추가
            # 최근 N일 데이터 생성 (1분 간격)
            timestamps = []
            for i in range(self.days_back * 24 * 60):
                timestamps.append(now - timedelta(minutes=i))
            
            # 데이터프레임 길이에 맞게 타임스탬프 조정
            if len(timestamps) > len(df):
                timestamps = timestamps[:len(df)]
            elif len(timestamps) < len(df):
                # 더 많은 타임스탬프 생성
                for i in range(len(df) - len(timestamps)):
                    timestamps.append(timestamps[-1] - timedelta(minutes=1))
            
            # 타임스탬프 열 추가
            df['timestamp'] = timestamps
            
            # 열 순서 변경 (timestamp를 첫 번째 열로)
            cols = df.columns.tolist()
            cols.remove('timestamp')
            cols = ['timestamp'] + cols
            df = df[cols]
            
            # 수정된 CSV 파일 저장
            output_file = os.path.join(self.output_dir, os.path.basename(csv_file))
            df.to_csv(output_file, index=False)
            
            print(f"    {output_file}에 저장되었습니다.")
            
        except Exception as e:
            print(f"    처리 중 오류 발생: {str(e)}")

# 메인 실행 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CSV 파일 형식 수정 시스템')
    parser.add_argument('--data-dir', type=str, default='data/live', help='원본 CSV 데이터 디렉토리')
    parser.add_argument('--output-dir', type=str, default='data/formatted', help='형식이 수정된 CSV 출력 디렉토리')
    parser.add_argument('--days-back', type=int, default=30, help='생성할 과거 데이터 일수')
    
    args = parser.parse_args()
    
    # CSV 형식 수정 시스템 초기화 및 실행
    formatter = CSVFormatter(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        days_back=args.days_back
    )
    
    formatter.run() 