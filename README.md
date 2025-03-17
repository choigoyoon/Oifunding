# 암호화폐 데이터 통합 관리 시스템

## 개요
- 암호화폐 거래소 데이터 수집 및 관리 시스템
- CCXT와 Coinalyze API를 활용한 데이터 수집
- 실시간 데이터 모니터링 및 저장

## 주요 기능
- 과거 데이터 누락 검사 및 복구
- 실시간 데이터 수집 및 저장
- 단일 CSV 파일에 시간순 정렬 저장
- 자동 심볼 추가 및 관리
- 안전한 종료 메커니즘
- 데이터 장기 보관
- 소수점 정확도 유지

## 설치 방법
```bash
git clone https://github.com/[사용자명]/crypto_data_manager.git
cd crypto_data_manager
pip install -r requirements.txt
```

## 사용 방법
```bash
python main.py
```

## 설정
- API 키 설정
- 데이터 저장 경로 설정
- 수집 주기 설정

## 라이선스
MIT License 