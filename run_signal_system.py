import os
import json
import argparse
import pandas as pd
from datetime import datetime
from collections import defaultdict
from get_all_signals import SignalExtractor

class SignalSharingSystem:
    """
    모든 암호화폐에서 최소 1개의 신호를 생성하고, 다수의 신호가 있는 경우 점수화하여 공유하는 시스템
    """
    
    def __init__(self, 
                 data_dir: str = 'data/live', 
                 results_dir: str = 'data/results',
                 output_dir: str = 'data/alarms',
                 share_dir: str = 'data/share',
                 initial_min_score: float = 70.0,
                 min_score_step: float = 5.0,
                 min_signals_per_symbol: int = 1,
                 top_signals_count: int = 10):
        """
        초기화
        
        Args:
            data_dir: CSV 데이터 디렉토리
            results_dir: JSON 결과 디렉토리
            output_dir: 알람 출력 디렉토리
            share_dir: 공유 파일 디렉토리
            initial_min_score: 초기 최소 가능성 점수 (기본값: 70.0)
            min_score_step: 점수 감소 단계 (기본값: 5.0)
            min_signals_per_symbol: 심볼당 최소 신호 수 (기본값: 1)
            top_signals_count: 상위 신호 개수 (기본값: 10)
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.share_dir = share_dir
        self.initial_min_score = initial_min_score
        self.min_score_step = min_score_step
        self.min_signals_per_symbol = min_signals_per_symbol
        self.top_signals_count = top_signals_count
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(share_dir, exist_ok=True)
        
        # 신호 추출기 초기화
        self.extractor = SignalExtractor(
            data_dir=data_dir,
            results_dir=results_dir,
            output_dir=output_dir,
            initial_min_score=initial_min_score,
            min_score_step=min_score_step,
            min_signals_per_symbol=min_signals_per_symbol
        )
        
        # 결과 저장 변수
        self.all_signals = []
        self.symbol_signals = {}
        self.symbols_without_signals = []
        self.top_signals = []
        self.symbol_rankings = []
    
    def run(self, analysis_file: str = None):
        """
        신호 시스템 실행
        
        Args:
            analysis_file: 분석 결과 파일 경로 (지정하지 않으면 가장 최근 파일 사용)
        """
        print("=" * 50)
        print("암호화폐 신호 시스템 실행")
        print("=" * 50)
        
        # 1. 적응형 임계값으로 모든 심볼에서 신호 추출
        print("\n1. 모든 심볼에서 신호 추출 중...")
        self.all_signals, self.symbol_signals = self.extractor.extract_signals_with_adaptive_threshold()
        self.symbols_without_signals = self.extractor.symbols_without_signals
        
        # 2. 결과 저장
        signals_file = os.path.join(self.output_dir, f'all_symbols_signals_{self.initial_min_score}.json')
        self.extractor.save_results(signals_file)
        
        # 3. 신호 점수화 및 랭킹
        print("\n2. 신호 점수화 및 랭킹 생성 중...")
        self._calculate_rankings()
        
        # 4. 공유 파일 생성
        print("\n3. 공유 파일 생성 중...")
        self._generate_sharing_files()
        
        # 5. 요약 출력
        self.extractor.print_summary()
        self._print_rankings()
    
    def _calculate_rankings(self):
        """
        신호 점수화 및 랭킹 계산
        """
        # 상위 신호 추출
        self.top_signals = sorted(self.all_signals, key=lambda x: x['score'], reverse=True)[:self.top_signals_count]
        
        # 심볼별 평균 점수 계산
        symbol_avg_scores = {}
        symbol_signal_counts = {}
        
        for symbol, signals in self.symbol_signals.items():
            if signals:
                total_score = sum(signal['score'] for signal in signals)
                avg_score = total_score / len(signals)
                symbol_avg_scores[symbol] = avg_score
                symbol_signal_counts[symbol] = len(signals)
        
        # 심볼 랭킹 생성 (평균 점수 기준)
        self.symbol_rankings = []
        for symbol, avg_score in sorted(symbol_avg_scores.items(), key=lambda x: x[1], reverse=True):
            self.symbol_rankings.append({
                'symbol': symbol,
                'avg_score': avg_score,
                'signal_count': symbol_signal_counts[symbol],
                'highest_score': max(signal['score'] for signal in self.symbol_signals[symbol]),
                'lowest_score': min(signal['score'] for signal in self.symbol_signals[symbol])
            })
    
    def _generate_sharing_files(self):
        """
        공유 파일 생성
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 상위 신호 공유 파일
        top_signals_file = os.path.join(self.share_dir, f'top_signals_{timestamp}.json')
        with open(top_signals_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'top_signals': self.top_signals,
                'description': f'상위 {self.top_signals_count}개 신호 (점수 기준)'
            }, f, indent=2)
        print(f"상위 신호 파일 생성: {top_signals_file}")
        
        # 2. 심볼 랭킹 공유 파일
        symbol_rankings_file = os.path.join(self.share_dir, f'symbol_rankings_{timestamp}.json')
        with open(symbol_rankings_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'symbol_rankings': self.symbol_rankings,
                'description': '심볼별 평균 점수 랭킹'
            }, f, indent=2)
        print(f"심볼 랭킹 파일 생성: {symbol_rankings_file}")
        
        # 3. 모든 심볼 신호 공유 파일 (심볼별로 최고 점수 신호 1개씩)
        all_symbols_best_signals = []
        for symbol, signals in self.symbol_signals.items():
            if signals:
                # 각 심볼의 최고 점수 신호 추출
                best_signal = max(signals, key=lambda x: x['score'])
                all_symbols_best_signals.append(best_signal)
        
        all_symbols_file = os.path.join(self.share_dir, f'all_symbols_best_signals_{timestamp}.json')
        with open(all_symbols_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'all_symbols_best_signals': sorted(all_symbols_best_signals, key=lambda x: x['score'], reverse=True),
                'description': '모든 심볼의 최고 점수 신호 1개씩'
            }, f, indent=2)
        print(f"모든 심볼 최고 신호 파일 생성: {all_symbols_file}")
        
        # 4. HTML 보고서 생성
        self._generate_html_report(timestamp)
    
    def _generate_html_report(self, timestamp):
        """
        HTML 보고서 생성
        
        Args:
            timestamp: 타임스탬프
        """
        html_file = os.path.join(self.share_dir, f'signal_report_{timestamp}.html')
        
        # HTML 템플릿
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>암호화폐 신호 보고서 - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .high-score {{ background-color: #d4edda; }}
                .medium-score {{ background-color: #fff3cd; }}
                .low-score {{ background-color: #f8d7da; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary {{ margin-bottom: 30px; }}
                .summary p {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>암호화폐 신호 보고서</h1>
                <p>생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="summary">
                    <h2>요약</h2>
                    <p>총 신호 수: {len(self.all_signals)}</p>
                    <p>신호가 있는 심볼 수: {len(self.symbol_signals)}</p>
                    <p>신호가 없는 심볼 수: {len(self.symbols_without_signals)}</p>
                    <p>사용된 최소 점수: {self.extractor.initial_min_score - (len(self.symbols_without_signals) > 0) * self.extractor.min_score_step * ((self.extractor.initial_min_score // self.extractor.min_score_step) + 1)}</p>
                </div>
                
                <h2>상위 {self.top_signals_count}개 신호</h2>
                <table>
                    <tr>
                        <th>순위</th>
                        <th>심볼</th>
                        <th>타임프레임</th>
                        <th>이벤트 유형</th>
                        <th>기간</th>
                        <th>점수</th>
                    </tr>
        """
        
        # 상위 신호 테이블 내용
        for i, signal in enumerate(self.top_signals):
            score_class = "high-score" if signal['score'] >= 90 else "medium-score" if signal['score'] >= 70 else "low-score"
            html_content += f"""
                    <tr class="{score_class}">
                        <td>{i+1}</td>
                        <td>{signal['symbol']}</td>
                        <td>{signal['timeframe']}</td>
                        <td>{signal['event_type']}</td>
                        <td>{signal['period']}</td>
                        <td>{signal['score']:.1f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h2>심볼 랭킹 (상위 20개)</h2>
                <table>
                    <tr>
                        <th>순위</th>
                        <th>심볼</th>
                        <th>평균 점수</th>
                        <th>신호 수</th>
                        <th>최고 점수</th>
                        <th>최저 점수</th>
                    </tr>
        """
        
        # 심볼 랭킹 테이블 내용
        for i, ranking in enumerate(self.symbol_rankings[:20]):
            score_class = "high-score" if ranking['avg_score'] >= 90 else "medium-score" if ranking['avg_score'] >= 70 else "low-score"
            html_content += f"""
                    <tr class="{score_class}">
                        <td>{i+1}</td>
                        <td>{ranking['symbol']}</td>
                        <td>{ranking['avg_score']:.1f}</td>
                        <td>{ranking['signal_count']}</td>
                        <td>{ranking['highest_score']:.1f}</td>
                        <td>{ranking['lowest_score']:.1f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h2>모든 심볼의 최고 점수 신호</h2>
                <table>
                    <tr>
                        <th>심볼</th>
                        <th>타임프레임</th>
                        <th>이벤트 유형</th>
                        <th>기간</th>
                        <th>점수</th>
                    </tr>
        """
        
        # 모든 심볼 최고 점수 신호 테이블 내용
        all_best_signals = []
        for symbol, signals in self.symbol_signals.items():
            if signals:
                best_signal = max(signals, key=lambda x: x['score'])
                all_best_signals.append(best_signal)
        
        for signal in sorted(all_best_signals, key=lambda x: x['score'], reverse=True):
            score_class = "high-score" if signal['score'] >= 90 else "medium-score" if signal['score'] >= 70 else "low-score"
            html_content += f"""
                    <tr class="{score_class}">
                        <td>{signal['symbol']}</td>
                        <td>{signal['timeframe']}</td>
                        <td>{signal['event_type']}</td>
                        <td>{signal['period']}</td>
                        <td>{signal['score']:.1f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML 보고서 생성: {html_file}")
    
    def _print_rankings(self):
        """
        랭킹 정보 출력
        """
        print("\n" + "=" * 50)
        print("신호 랭킹 정보")
        print("=" * 50)
        
        print(f"\n상위 {self.top_signals_count}개 신호:")
        for i, signal in enumerate(self.top_signals):
            print(f"  {i+1}. {signal['symbol']} - {signal['timeframe']} - {signal['event_type']} - {signal['period']} - 점수: {signal['score']:.1f}")
        
        print("\n심볼 랭킹 (상위 10개):")
        for i, ranking in enumerate(self.symbol_rankings[:10]):
            print(f"  {i+1}. {ranking['symbol']} - 평균 점수: {ranking['avg_score']:.1f} - 신호 수: {ranking['signal_count']}")

# 메인 실행 코드
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='암호화폐 신호 시스템 실행')
    parser.add_argument('--data-dir', type=str, default='data/live', help='CSV 데이터 디렉토리')
    parser.add_argument('--results-dir', type=str, default='data/results', help='JSON 결과 디렉토리')
    parser.add_argument('--output-dir', type=str, default='data/alarms', help='알람 출력 디렉토리')
    parser.add_argument('--share-dir', type=str, default='data/share', help='공유 파일 디렉토리')
    parser.add_argument('--initial-min-score', type=float, default=70.0, help='초기 최소 가능성 점수')
    parser.add_argument('--min-score-step', type=float, default=5.0, help='점수 감소 단계')
    parser.add_argument('--min-signals', type=int, default=1, help='심볼당 최소 신호 수')
    parser.add_argument('--top-signals', type=int, default=10, help='상위 신호 개수')
    parser.add_argument('--analysis-file', type=str, help='분석 결과 파일 경로 (지정하지 않으면 가장 최근 파일 사용)')
    
    args = parser.parse_args()
    
    # 신호 공유 시스템 초기화 및 실행
    system = SignalSharingSystem(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        share_dir=args.share_dir,
        initial_min_score=args.initial_min_score,
        min_score_step=args.min_score_step,
        min_signals_per_symbol=args.min_signals,
        top_signals_count=args.top_signals
    )
    
    system.run(args.analysis_file) 