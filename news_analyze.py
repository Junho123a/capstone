import os
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
import seaborn as sns
from train_model_dLinear_finalVersion import FusionModel
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 운영체제별 기본 한글 폰트 설정
if platform.system() == 'Windows':
    # Windows 기본 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    # Linux 환경 (예: Ubuntu)
    # 설치된 한글 폰트 확인 후 설정 (예: NanumGothic)
    plt.rcParams['font.family'] = 'NanumGothic'

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# Define stock list globally
STOCK_LIST = [
    {"name": "삼성전자", "code": "005930"},
    {"name": "SK하이닉스", "code": "000660"},
    {"name": "현대차", "code": "005380"},
    {"name": "기아", "code": "000270"},
    {"name": "LIG넥스원", "code": "079550"},
    {"name": "삼성SDI", "code": "006400"},
    {"name": "셀트리온", "code": "068270"},
    {"name": "삼성바이오로직스", "code": "207940"},
    {"name": "KB금융", "code": "105560"},
    {"name": "신한지주", "code": "055550"},
    {"name": "LG전자", "code": "066570"},
    {"name": "아모레퍼시픽", "code": "090430"},
    {"name": "대한항공", "code": "003490"},
    {"name": "크래프톤", "code": "259960"},
    {"name": "삼성물산", "code": "028260"},
    {"name": "대우건설", "code": "047040"},
    {"name": "HMM", "code": "011200"},
    {"name": "한화솔루션", "code": "009830"},
    {"name": "NAVER", "code": "035420"},
    {"name": "KT", "code": "030200"}
]

class NewsImpactXAI:
    def __init__(self, model_dir, srtnCd, device='cpu'):
        self.model_dir = model_dir
        self.srtnCd = srtnCd
        self.device = torch.device(device)
        
        # Get stock name from code
        self.stock_name = self.get_stock_name_from_code(srtnCd)
        if not self.stock_name:
            raise ValueError(f"Stock code {srtnCd} not found in stock list")
        
        # 기본 디렉토리 설정
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'CSV'))
        self.news_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'news_CSV'))
        
        # 디렉토리가 없으면 생성
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.news_dir, exist_ok=True)
        
        # 기본 파라미터 설정
        self.sequence_length = 10
        self.prediction_length = 5
        self.scaler = None
        
        # 파라미터 및 모델 로드 시도
        try:
            self.load_parameters()
            self.load_model()
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            print("기본 모델 파라미터로 진행합니다.")
            self._create_mock_model()
    
    def _create_mock_model(self):
        """모델 파일이 없을 경우 모의 모델 생성"""
        self.sequence_length = 10
        self.prediction_length = 5
        
        # 간단한 스케일러 생성
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        
        # 모의 모델 생성
        self.model = FusionModel(
            seq_len=self.sequence_length,
            pred_len=self.prediction_length,
            enc_in=2  # 기본값으로 특성 2개만 가정
        )
        self.model.to(self.device)
        self.model.eval()
    
    def get_stock_name_from_code(self, code):
        """Convert stock code to stock name using the stock list"""
        for stock in STOCK_LIST:
            if stock["code"] == code:
                return stock["name"]
        return None
    
    def load_parameters(self):
        # 모델 디렉토리 확인
        model_folder = os.path.join(self.model_dir, f"N{self.srtnCd}")
        if not os.path.exists(model_folder):
            raise FileNotFoundError(f"모델 폴더가 존재하지 않습니다: {model_folder}")
        
        # 기간 정보 파일 로드
        period_file = os.path.join(model_folder, 'period_info.txt')
        if not os.path.exists(period_file):
            raise FileNotFoundError(f"기간 정보 파일이 존재하지 않습니다: {period_file}")
            
        with open(period_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('sequence_length'):
                    self.sequence_length = int(line.split('=')[1].strip())
                elif line.startswith('period'):
                    self.prediction_length = int(line.strip().split('=')[1])
        
        # 스케일러 로드
        scaler_path = os.path.join(model_folder, 'dlinear_scaler.joblib')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"스케일러 파일이 존재하지 않습니다: {scaler_path}")
            
        self.scaler = joblib.load(scaler_path)
    
    def load_model(self):
        # 모델 파일 경로
        model_path = os.path.join(self.model_dir, f"N{self.srtnCd}", 'dlinear_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
        
        # 특성 차원 가져오기
        feature_dim = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else 2
        
        # FusionModel 객체 생성
        self.model = FusionModel(
            seq_len=self.sequence_length,
            pred_len=self.prediction_length,
            enc_in=feature_dim,
            kernel_size=25
        )
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            print(f"모델 로딩 실패: {str(e)}")
            print("기본 모델로 진행합니다.")
            
        self.model.to(self.device)
        self.model.eval()  # 추론 모드로 설정

    def get_news_basename_from_pricefile(self):
        # 뉴스 파일명 생성
        return f"{self.stock_name}_20200101_20241231.csv"

    def load_price_data(self, target_date):
        # CSV 파일에서 정수형 날짜 처리
        file_path = os.path.join(self.base_dir, f"{self.stock_name}_20200101_20241231_KOSPI.csv")
        
        # 정수형 날짜를 문자열로 읽고 datetime 변환
        df = pd.read_csv(file_path, dtype={'basDt': str})
        df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d', errors='coerce')
        
        # 유효하지 않은 날짜 제거
        df = df.dropna(subset=['basDt'])
        
        # 날짜 범위 필터링 (datetime.date로 통일)
        start_date = target_date - timedelta(days=self.sequence_length*3)
        end_date = target_date + timedelta(days=1)
        df = df[(df['basDt'].dt.date >= start_date.date()) & 
                (df['basDt'].dt.date <= end_date.date())]
        
        return df.sort_values('basDt')
    
    def _create_mock_price_data(self, start_date, end_date):
        """테스트용 가상 주가 데이터 생성"""
        # 날짜 범위 생성 (주말 제외)
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            # 주말(5:토요일, 6:일요일) 제외
            if current_date.weekday() < 5:
                date_range.append(current_date)
            current_date += timedelta(days=1)
        
        # 가상 주가 데이터 생성
        mock_data = {
            'basDt': date_range,
            'clpr': np.random.normal(50000, 1000, len(date_range)),  # 종가
            'trqu': np.random.randint(100000, 1000000, len(date_range)),  # 거래량
            'mkp': np.random.normal(50000, 500, len(date_range))  # 시장가격
        }
        
        return pd.DataFrame(mock_data)

    def load_news_data(self, target_date):
        # 뉴스 파일 경로
        news_file = os.path.join(self.news_dir, f"{self.stock_name}_20200101_20241231.csv")
        
        try:
            # CSV 읽기 (날짜 컬럼을 문자열로 처리)
            df_news = pd.read_csv(news_file, dtype={'날짜': str})
            
            # 정수형 날짜(YYYYMMDD) -> datetime 변환
            df_news['basDt'] = pd.to_datetime(
                df_news['날짜'].astype(str).str[:8],  # 8자리로 제한
                format='%Y%m%d', 
                errors='coerce'
            )
            
            # 변환 실패한 행 제거
            df_news = df_news.dropna(subset=['basDt'])
            
            # 타겟 날짜 필터링
            target_date = target_date.date() if isinstance(target_date, datetime) else target_date
            df_news = df_news[df_news['basDt'].dt.date == target_date]
            
            # news_text 컬럼 추가 (제목 복사)
            df_news['news_text'] = df_news['제목']
            
            # 필요한 컬럼만 선택
            return df_news[['제목', '언론사', 'basDt', '링크', 'news_text']]
        except Exception as e:
            # 파일이 없거나 읽을 수 없는 경우
            print(f"뉴스 파일 읽기 실패: {str(e)}")
            return pd.DataFrame(columns=['제목', '언론사', 'basDt', '링크', 'news_text'])
    
    def _create_neutral_news_data(self, target_date):
        """중립적인 더미 뉴스 데이터 생성 (어텐션 가중치가 0에 가깝게 나오도록)"""
        neutral_news = [
            f"{self.stock_name}에 관한 특별한 뉴스가 없습니다.",
            f"{target_date.strftime('%Y-%m-%d')} 날짜에는 관련 뉴스가 없습니다.",
            f"시장 평균과 유사한 흐름을 보였습니다.",
            f"특이사항 없음",
            f"일반적인 거래 패턴 유지"
        ]
        
        # 가상 뉴스 데이터 생성
        mock_data = {
            'basDt': [target_date] * len(neutral_news),
            'news_text': neutral_news,
            '제목': neutral_news,
            '언론사': ['자동 생성'] * len(neutral_news),
            '링크': [''] * len(neutral_news)
        }
        
        return pd.DataFrame(mock_data)

    def prepare_data(self, price_df, target_date):
        # 대상 날짜를 date 객체로 변환
        target_date = target_date.date() if isinstance(target_date, datetime) else target_date
        
        # 대상 날짜 인덱스 찾기
        target_rows = price_df[price_df['basDt'].dt.date == target_date]
        
        if target_rows.empty:
            # 가장 가까운 이전 거래일 탐색
            valid_dates = price_df[price_df['basDt'].dt.date < target_date]
            if valid_dates.empty:
                raise ValueError("과거 거래 데이터가 존재하지 않음")
            nearest_date = valid_dates.iloc[-1]['basDt'].date()
            print(f"{target_date} 데이터 없음. 가장 가까운 거래일 {nearest_date} 사용")
            target_rows = price_df[price_df['basDt'].dt.date == nearest_date]
        
        # 특성 추출 및 전처리
        features = ['clpr', 'trqu', 'mkp'][:self.scaler.n_features_in_]
        X = price_df[features].values
        X_scaled = self.scaler.transform(X)
        
        # 시퀀스 생성
        target_idx = target_rows.index[0]
        seq_start = max(0, target_idx - self.sequence_length + 1)
        sequence = X_scaled[seq_start:target_idx+1]
        
        # 패딩 처리
        if len(sequence) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(sequence), X_scaled.shape[1]))
            sequence = np.vstack([padding, sequence])
        
        # 텐서 변환 [batch, seq_len, features]
        return torch.FloatTensor(sequence).unsqueeze(0)

    def analyze_news_impact(self, stock_seq, news_texts):
        # 뉴스가 없는 경우의 처리
        if not news_texts:
            print("분석할 뉴스가 없습니다.")
            return []
        
        # 뉴스가 없는 경우의 예측 (기준선)
        with torch.no_grad():
            try:
                baseline_pred, baseline_attn = self.model(stock_seq.to(self.device), [""])
                baseline_pred = baseline_pred.cpu().numpy().flatten()
            except Exception as e:
                print(f"기준선 예측 오류: {str(e)}")
                # 더미 예측값 생성
                baseline_pred = np.zeros(self.prediction_length)
                baseline_attn = torch.tensor([[0.0]])
        
        news_impacts = []
        
        for news_text in news_texts:
            # 뉴스가 있는 경우의 예측
            with torch.no_grad():
                try:
                    news_pred, attention_weight = self.model(stock_seq.to(self.device), [news_text])
                    news_pred = news_pred.cpu().numpy().flatten()
                except Exception as e:
                    print(f"뉴스 '{news_text}' 예측 오류: {str(e)}")
                    # 더미 예측값 생성
                    news_pred = baseline_pred + np.random.normal(0, 0.01, size=baseline_pred.shape)
                    attention_weight = torch.tensor([[0.0]])
            
            # 뉴스 영향 계산 (뉴스 유무에 따른 예측 차이)
            impact = news_pred - baseline_pred
            impact_score = np.abs(impact).sum()  # 영향도 점수
            impact_direction = "상승" if impact.mean() > 0 else "하락"
            
            news_impacts.append({
                'news_text': news_text,
                'attention_weight': attention_weight.item(),
                'prediction': news_pred,
                'baseline': baseline_pred,
                'impact': impact,
                'impact_score': impact_score,
                'direction': impact_direction
            })
        
        # 영향도 점수 기준으로 정렬
        sorted_news = sorted(news_impacts, key=lambda x: x['impact_score'], reverse=True)
        
        return sorted_news

    def visualize_news_impact(self, news_impacts, top_n=5):
        if not news_impacts:
            print("시각화할 데이터가 없습니다.")
            return
        
        # 상위 N개 뉴스만 선택
        top_news = news_impacts[:min(top_n, len(news_impacts))]
        
        # 시각화 1: 영향력 점수
        plt.figure(figsize=(12, 6))
        
        news_texts = [f"{i+1}. {news['news_text'][:30]}..." for i, news in enumerate(top_news)]
        impact_scores = [news['impact_score'] for news in top_news]
        impact_directions = [news['direction'] for news in top_news]
        
        # 영향 방향에 따른 색상 설정
        colors = ['#ff9999' if direction == '하락' else '#99ff99' for direction in impact_directions]
        
        # 가로 막대 그래프
        bars = plt.barh(news_texts, impact_scores, color=colors)
        plt.xlabel('뉴스 영향력 점수 (예측 변화량 절대값 합)')
        plt.title(f'{self.srtnCd} ({self.stock_name}) - 상위 {len(top_news)} 뉴스 영향력')
        
        # 상승/하락 설명 범례
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#99ff99', label='상승 영향'),
            Patch(facecolor='#ff9999', label='하락 영향')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # 영향력 값 표시
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
        
        # 시각화 2: 주가 예측 비교
        plt.figure(figsize=(12, 8))
        
        days = list(range(1, len(top_news[0]['baseline']) + 1))
        
        # 기준선(뉴스 없음)
        plt.plot(days, top_news[0]['baseline'], 'k--', label='뉴스 없음 (기준)')
        
        # 각 뉴스에 대한 예측
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_news)))
        for i, news in enumerate(top_news):
            plt.plot(days, news['prediction'], color=colors[i], 
                     label=f"{i+1}. {news['news_text'][:20]}... ({news['direction']})")
        
        plt.xlabel('예측 기간 (일)')
        plt.ylabel('예측 주가 (스케일링됨)')
        plt.title(f'{self.srtnCd} ({self.stock_name}) - 뉴스별 주가 예측 비교')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        
        # 시각화 3: 어텐션 가중치 비교
        plt.figure(figsize=(12, 6))
        attention_weights = [news['attention_weight'] for news in top_news]
        
        # 가로 막대 그래프
        bars = plt.barh(news_texts, attention_weights, color='skyblue')
        plt.xlabel('모델 어텐션 가중치')
        plt.title(f'{self.srtnCd} ({self.stock_name}) - 뉴스별 어텐션 가중치')
        
        # 가중치 값 표시
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()

    def run_xai_analysis(self, target_date):
        print(f"=== {self.srtnCd} ({self.stock_name}) 종목에 대한 뉴스 영향 XAI 분석 ===")
        print(f"분석 날짜: {target_date.strftime('%Y-%m-%d')}")
        
        # 주가 데이터 로드 (타겟 날짜 기준)
        try:
            price_df = self.load_price_data(target_date)
            if price_df.empty:
                print("주가 데이터를 로드할 수 없습니다.")
                return
        except Exception as e:
            print(f"주가 데이터 로드 중 오류 발생: {str(e)}")
            return
        
        # 뉴스 데이터 로드 (타겟 날짜만)
        try:
            news_df = self.load_news_data(target_date)
            if news_df.empty:
                print(f"{target_date.strftime('%Y-%m-%d')}에 해당하는 뉴스가 없습니다.")
                
                # 중립적인 더미 뉴스 데이터 생성해서 진행
                news_df = self._create_neutral_news_data(target_date)
                print("분석을 위한 중립적인 더미 뉴스 데이터를 생성합니다.")
        except Exception as e:
            print(f"뉴스 데이터 로드 중 오류 발생: {str(e)}")
            news_df = self._create_neutral_news_data(target_date)
            print("뉴스 데이터 로드 오류로 인한 중립적인 더미 뉴스 데이터를 생성합니다.")
        
        # 주가 데이터 준비
        try:
            stock_seq = self.prepare_data(price_df, target_date)
        except Exception as e:
            print(f"데이터 준비 중 오류 발생: {str(e)}")
            return
        
        # 모든 뉴스 텍스트 목록 생성
        news_texts = news_df['news_text'].tolist()
        
        print(f"{target_date.strftime('%Y-%m-%d')}일자 뉴스 {len(news_texts)}개에 대한 분석을 시작합니다.")
        
        # 각 뉴스별 영향 분석
        try:
            news_impacts = self.analyze_news_impact(stock_seq, news_texts)
            if not news_impacts:
                print("뉴스 영향 분석 결과가 없습니다.")
                return
        except Exception as e:
            print(f"뉴스 영향 분석 중 오류 발생: {str(e)}")
            return
        
        # 뉴스 영향 시각화
        try:
            self.visualize_news_impact(news_impacts)
        except Exception as e:
            print(f"시각화 중 오류 발생: {str(e)}")
        
        # 최고 영향력 뉴스 추출
        top_news = news_impacts[0]
        
        print("\n=== 최종 분석 결과 ===")
        print(f"가장 높은 영향력을 가진 뉴스:")
        print(f"  \"{top_news['news_text']}\"")
        print(f"  영향력 점수: {top_news['impact_score']:.4f}")
        print(f"  영향 방향: {top_news['direction']}")
        print(f"  어텐션 가중치: {top_news['attention_weight']:.4f}")
        
        # 예측 결과 해석
        pred_changes = top_news['prediction']
        pred_direction = "상승" if pred_changes[0] > 0 else "하락"
        
        print(f"\n향후 {self.prediction_length}일간 주가 예측:")
        print(f"  전체 추세: {pred_direction} 예상")
        for i, change in enumerate(pred_changes):
            print(f"  {i+1}일 후: {change:.4f}")
        
        # 전략적 조언
        if pred_direction == "상승":
            print("\n투자 조언: 상승세가 예상되므로 매수 검토를 고려해볼 수 있습니다.")
        else:
            print("\n투자 조언: 하락세가 예상되므로 추가 매수는 신중하게 검토하세요.")

def display_available_stocks():
    """Display all available stocks with their codes and names"""
    print("\n=== 사용 가능한 종목 목록 ===")
    print(f"{'종목코드':<10} {'종목명':<20}")
    print("-" * 30)
    for stock in STOCK_LIST:
        print(f"{stock['code']:<10} {stock['name']:<20}")
    print("-" * 30)

def main():
    print("주가 뉴스 영향 분석 시스템 (v1.0)")
    print("=" * 50)
    
    # Display available commands
    print("\n명령어 목록:")
    print("  list : 사용 가능한 종목 목록 표시")
    print("  quit : 프로그램 종료")
    
    while True:
        # 사용자 입력 받기
        srtnCd = input("\n종목 코드를 입력하세요 (list:목록 표시, quit:종료): ")
        
        if srtnCd.lower() == 'quit':
            print("프로그램을 종료합니다.")
            break
        
        if srtnCd.lower() == 'list':
            display_available_stocks()
            continue
        
        if not srtnCd:
            print("종목 코드는 필수입니다.")
            continue
        
        # Check if stock code exists in the list
        stock_name = None
        for stock in STOCK_LIST:
            if stock["code"] == srtnCd:
                stock_name = stock["name"]
                break
                
        if not stock_name:
            print(f"입력한 종목 코드 {srtnCd}를 찾을 수 없습니다. 유효한 종목 코드를 입력하세요.")
            continue
        
        # 단일 날짜 분석
        target_date_str = input("분석 날짜를 입력하세요 (YYYYMMDD, 기본값: 오늘): ")
        
        try:
            if target_date_str:
                target_date = datetime.strptime(target_date_str, '%Y%m%d')
            else:
                # 기본값: 오늘 날짜
                target_date = datetime.now()
        except ValueError as e:
            print(f"날짜 형식 오류: {e}. YYYYMMDD 형식으로 입력하세요.")
            continue
        
        # 모델 디렉토리
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
        os.makedirs(model_dir, exist_ok=True)
        
        # 뉴스 영향 XAI 클래스 초기화
        try:
            xai = NewsImpactXAI(model_dir, srtnCd)
            
            # 단일 날짜 XAI 분석 실행
            xai.run_xai_analysis(target_date)
            
            # Ask if user wants to analyze another stock
            choice = input("\n다른 종목을 분석하시겠습니까? (y/n): ")
            if choice.lower() != 'y':
                print("프로그램을 종료합니다.")
                break
        except Exception as e:
            print(f"분석 중 오류가 발생했습니다: {str(e)}")
            continue

if __name__ == "__main__":
    main()