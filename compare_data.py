import pandas as pd
import json
import os
from datetime import datetime
import random

# 분석할 심볼 목록 (주요 심볼 + 랜덤 선택된 심볼)
main_symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK']

# 결과 디렉토리에서 모든 심볼 가져오기
all_symbols = [f.split('_')[0] for f in os.listdir('data/results') if f.endswith('_changes.json')]
random_symbols = random.sample([s for s in all_symbols if s not in main_symbols], 7)
symbols = main_symbols + random_symbols

print("=" * 50)
print("CSV 파일과 JSON 분석 결과 비교")
print("=" * 50)

# 결과 요약을 위한 데이터 수집
summary_data = {
    'total_symbols': 0,
    'total_rows': 0,
    'interval_counts': {},
    'missing_data': {},
    'timeframes': set(),
    'oi_changes_avg': 0,
    'funding_changes_avg': 0,
    'volume_spikes_avg': 0
}

for symbol in symbols:
    csv_path = f'data/live/{symbol}.csv'
    json_path = f'data/results/{symbol}_changes.json'
    
    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        print(f"{symbol} 파일이 존재하지 않습니다.")
        continue
    
    summary_data['total_symbols'] += 1
    
    # CSV 파일 분석
    df = pd.read_csv(csv_path)
    summary_data['total_rows'] += len(df)
    
    # JSON 파일 분석
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    print(f"\n{symbol} 데이터 분석:")
    print("-" * 30)
    
    # CSV 파일 정보
    print(f"CSV 파일 행 수: {len(df)}")
    print(f"CSV 파일 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"CSV 파일 컬럼: {', '.join(df.columns)}")
    
    # 시간 간격 분석
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    time_diffs = df['datetime'].diff().dropna()
    
    # 시간 간격 빈도 계산
    time_diff_minutes = time_diffs.dt.total_seconds() / 60
    interval_counts = time_diff_minutes.value_counts().sort_index()
    
    print("\n시간 간격 분포:")
    for interval, count in interval_counts.items():
        print(f"  {interval}분: {count}회 ({count/len(time_diff_minutes)*100:.1f}%)")
        
        # 요약 데이터에 추가
        if interval not in summary_data['interval_counts']:
            summary_data['interval_counts'][interval] = 0
        summary_data['interval_counts'][interval] += count
    
    # JSON 파일 정보
    timeframes = list(json_data['results'].keys())
    print(f"\nJSON 분석 결과 타임프레임: {timeframes}")
    
    for tf in timeframes:
        summary_data['timeframes'].add(tf)
    
    # 1분 타임프레임 정보
    if '1M' in json_data['results']:
        oi_changes = json_data['results']['1M'].get('oi_changes', [])
        funding_changes = json_data['results']['1M'].get('funding_changes', [])
        volume_spikes = json_data['results']['1M'].get('volume_spikes', [])
        
        print(f"1분 타임프레임 OI 변화 수: {len(oi_changes)}")
        print(f"1분 타임프레임 자금 비율 변화 수: {len(funding_changes)}")
        print(f"1분 타임프레임 거래량 스파이크 수: {len(volume_spikes)}")
        
        summary_data['oi_changes_avg'] += len(oi_changes)
        summary_data['funding_changes_avg'] += len(funding_changes)
        summary_data['volume_spikes_avg'] += len(volume_spikes)
    
    # 누락된 데이터 확인
    possible_columns = ['open_interest', 'funding_rate', 'predicted_funding_rate', 'long_percentage', 'short_percentage']
    missing_columns = []
    
    for col in possible_columns:
        if col in df.columns and df[col].isna().sum() > 0:
            missing_count = df[col].isna().sum()
            missing_columns.append((col, missing_count))
            
            # 요약 데이터에 추가
            if col not in summary_data['missing_data']:
                summary_data['missing_data'][col] = {'count': 0, 'symbols': 0}
            summary_data['missing_data'][col]['count'] += missing_count
            summary_data['missing_data'][col]['symbols'] += 1
    
    if missing_columns:
        print("\n누락된 데이터가 있는 컬럼:")
        for col, missing_count in missing_columns:
            print(f"  {col}: {missing_count}개 ({missing_count/len(df)*100:.1f}%)")
    else:
        print("\n모든 데이터가 완전합니다.")

# 요약 정보 출력
print("\n" + "=" * 50)
print("데이터 분석 요약")
print("=" * 50)

print(f"분석된 심볼 수: {summary_data['total_symbols']}")
print(f"총 데이터 행 수: {summary_data['total_rows']}")
print(f"평균 데이터 행 수: {summary_data['total_rows'] / summary_data['total_symbols']:.1f}")

print("\n시간 간격 분포 (전체):")
total_intervals = sum(summary_data['interval_counts'].values())
for interval, count in sorted(summary_data['interval_counts'].items()):
    print(f"  {interval}분: {count}회 ({count/total_intervals*100:.1f}%)")

print("\n타임프레임 지원:")
for tf in sorted(summary_data['timeframes']):
    print(f"  {tf}")

print("\n1분 타임프레임 평균 이벤트 수:")
print(f"  OI 변화: {summary_data['oi_changes_avg'] / summary_data['total_symbols']:.1f}")
print(f"  자금 비율 변화: {summary_data['funding_changes_avg'] / summary_data['total_symbols']:.1f}")
print(f"  거래량 스파이크: {summary_data['volume_spikes_avg'] / summary_data['total_symbols']:.1f}")

print("\n누락된 데이터 요약:")
for col, data in summary_data['missing_data'].items():
    print(f"  {col}: {data['count']}개 누락 ({data['symbols']}개 심볼에서 발생)")

print("\n" + "=" * 50) 