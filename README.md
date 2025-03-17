# 암호화폐 데이터 통합 관리 시스템

암호화폐 거래소의 다양한 데이터를 수집하고 관리하는 시스템입니다.

## 개요
암호화폐 거래소의 데이터를 수집하고 관리하는 통합 시스템입니다.

## 주요 기능
- CCXT를 통한 과거 데이터 수집 및 보완
- 실시간 데이터 수집 및 저장
- 데이터 백업 및 아카이브
- 안전한 종료 메커니즘
- 자동 심볼 관리
- 소수점 정밀도 보존

## 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/choigoyoon/Oifunding.git
cd Oifunding
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 설정 파일 구성:
- `src/config.py` 파일에서 API 키와 설정을 구성합니다.

2. 시스템 실행:
```bash
python -m src.main
```

## 프로젝트 구조

```
crypto_data_manager/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── api_key_manager.py
│   │   ├── ccxt_manager.py
│   │   └── request_manager.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── processor.py
│   └── storage/
│       ├── __init__.py
│       └── data_manager.py
├── data/
│   ├── live/
│   ├── backup/
│   ├── archive/
│   └── long_term/
├── logs/
├── requirements.txt
└── README.md
```

## 데이터 저장 구조

- `data/live/`: 실시간 데이터 저장
- `data/backup/`: 데이터 백업
- `data/archive/`: 오래된 데이터 아카이브
- `data/long_term/`: 장기 데이터 저장

## 로깅

- `logs/` 디렉토리에 로그 파일이 저장됩니다.
- 로그 레벨: INFO, WARNING, ERROR

## 라이선스

MIT License 