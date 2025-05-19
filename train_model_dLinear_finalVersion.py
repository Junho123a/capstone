import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import datetime
warnings.filterwarnings('ignore')

class MovingAvg(nn.Module):
    """
    추세 컴포넌트를 추출하기 위한 이동 평균 블록
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 시계열 양쪽 끝에 패딩 추가
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x

class SeriesDecomp(nn.Module):
    """
    시계열 분해 블록
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        trend = moving_mean
        seasonal = x - trend
        return trend, seasonal

class DLinear(nn.Module):
    """
    시계열 예측을 위한 DLinear 모델
    """
    def __init__(self, seq_len, pred_len, enc_in, kernel_size=25):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 분해 커널 사이즈
        self.decomposition = SeriesDecomp(kernel_size)
        
        # 각 특성별 추세 및 계절성을 위한 선형 레이어
        self.linear_trends = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(enc_in)])
        self.linear_seasonals = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(enc_in)])

    def forward(self, x):
        # x: [Batch, seq_len, enc_in]
        batch_size = x.size(0)
        enc_in = x.size(2)
        
        # 추세 및 계절성 추출
        trend_init, seasonal_init = self.decomposition(x)
        
        # 출력 초기화
        trend_output = []
        seasonal_output = []
        
        # 각 특성별로 처리
        for i in range(enc_in):
            # i번째 특성 추출
            trend_data = trend_init[:, :, i]
            seasonal_data = seasonal_init[:, :, i]
            
            # 선형 레이어 적용
            trend_result = self.linear_trends[i](trend_data)
            seasonal_result = self.linear_seasonals[i](seasonal_data)
            
            # 출력에 추가
            trend_output.append(trend_result.unsqueeze(-1))
            seasonal_output.append(seasonal_result.unsqueeze(-1))
        
        # 결과 결합
        trend_output = torch.cat(trend_output, dim=-1)
        seasonal_output = torch.cat(seasonal_output, dim=-1)
        
        # [batch_size, pred_len, enc_in] 형태로 변환
        trend_output = trend_output.unsqueeze(1) if trend_output.dim() == 2 else trend_output
        seasonal_output = seasonal_output.unsqueeze(1) if seasonal_output.dim() == 2 else seasonal_output
        
        # 최종 예측
        x_out = trend_output + seasonal_output
        
        return x_out

class StockNewsDataset(Dataset):
    """
    주가 및 뉴스 데이터셋
    """
    def __init__(self, stock_data, news_dict, dates, seq_len, pred_len):
        self.stock_data = stock_data  # [N, feature_dim]
        self.news_dict = news_dict    # 날짜와 뉴스 헤드라인을 매핑하는 딕셔너리
        self.dates = dates            # stock_data에 해당하는 날짜 배열
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 유효한 인덱스 계산
        self.indices = list(range(len(stock_data) - seq_len - pred_len + 1))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        
        # 시퀀스에 대한 주가 데이터 추출
        stock_seq = self.stock_data[i:i+self.seq_len]
        stock_target = self.stock_data[i+self.seq_len:i+self.seq_len+self.pred_len, 0]  # 첫 번째 특성이 타겟(종가)
        
        # 시퀀스의 마지막 날에 대한 뉴스 추출
        seq_end_date = self.dates[i+self.seq_len-1]
        news_text = self.news_dict.get(seq_end_date, "")
        
        return torch.FloatTensor(stock_seq), torch.FloatTensor(stock_target), news_text

class FusionModel(nn.Module):
    """
    koBERT와 DLinear를 결합한 융합 모델 (어텐션 기반 XAI 포함)
    """
    def __init__(self, seq_len, pred_len, enc_in, kernel_size=25):
        super(FusionModel, self).__init__()
        # koBERT 인코더
        self.tokenizer = AutoTokenizer.from_pretrained(
            "monologg/kobert", 
            trust_remote_code=True  # 필수 추가
        )
        self.bert_model = AutoModel.from_pretrained(
            "monologg/kobert",
            trust_remote_code=True  # 필수 추가
        )
        
        self.bert_hidden_size = self.bert_model.config.hidden_size
        
        # 주가를 위한 DLinear
        self.dlinear = DLinear(seq_len, pred_len, enc_in, kernel_size)
        
        # 융합 레이어
        fusion_input_size = self.bert_hidden_size + pred_len * enc_in
        self.fusion = nn.Linear(fusion_input_size, 128)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len)
        )
        
        # XAI를 위한 어텐션
        self.attention = nn.Sequential(
            nn.Linear(fusion_input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def encode_news(self, news_texts):
        """
        koBERT를 사용하여 뉴스 텍스트 인코딩
        """
        # 빈 뉴스 텍스트 처리
        if not news_texts or all(not text for text in news_texts):
            # 뉴스가 없는 경우 0 텐서 반환
            device = next(self.parameters()).device
            return torch.zeros(len(news_texts) if news_texts else 1, self.bert_hidden_size, device=device)
        
        # 뉴스 헤드라인 토큰화
        encoded_input = self.tokenizer(news_texts, padding=True, truncation=True, 
                                      return_tensors="pt", max_length=128)
        
        # 텐서를 모델과 같은 디바이스로 이동
        device = next(self.parameters()).device
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        
        # BERT 출력 얻기
        with torch.no_grad():  # 메모리 절약을 위해 BERT에 대한 그라디언트는 추적하지 않음
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # CLS 토큰 표현을 문장 임베딩으로 사용
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
    
    def forward(self, stock_data, news_texts):
        # DLinear로 주가 데이터 처리
        dlinear_output = self.dlinear(stock_data)  # [batch_size, pred_len, enc_in]
        batch_size = dlinear_output.size(0)
        dlinear_flat = dlinear_output.reshape(batch_size, -1)  # [batch_size, pred_len * enc_in]
        
        # koBERT로 뉴스 처리
        bert_output = self.encode_news(news_texts)  # [batch_size, bert_hidden_size]
        
        # 융합
        concat = torch.cat([dlinear_flat, bert_output], dim=1)  # [batch_size, pred_len * enc_in + bert_hidden_size]
        
        # XAI를 위한 어텐션 가중치 계산
        attention_scores = self.attention(concat)  # [batch_size, 1]
        attention_weights = torch.softmax(attention_scores, dim=0)  # 배치 전체에서 정규화
        
        # 융합 및 MLP 적용
        fusion_output = self.fusion(concat)  # [batch_size, 128]
        fusion_output = torch.relu(fusion_output)
        mlp_output = self.mlp(fusion_output)  # [batch_size, pred_len]
        
        return mlp_output, attention_weights

class RoboAdvisor:
    def __init__(self, sequence_length=60, prediction_length=5, batch_size=32, epochs=50, learning_rate=0.001, start_date=None, end_date=None):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # 날짜 범위 설정
        self.start_date = start_date
        self.end_date = end_date
        
        # 디렉토리 설정
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'CSV'))
        self.news_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'news_CSV'))
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 디바이스 설정
        self.device = torch.device('cpu')  # CPU 환경에서 실행
        print(f"사용 디바이스: {self.device}")
    
    def get_news_basename_from_pricefile(self, pricefile):
        """
        주가 파일 이름에서 뉴스 파일 이름 추출
        """
        parts = pricefile.split('_')
        if len(parts) >= 3:
            name = parts[0]
            return f"{name}_20200101_20241231.csv"
        else:
            return None
    
    def load_price_data(self, file_path):
        """
        주가 데이터 로드 및 전처리, 지정된 날짜 범위만 필터링
        """
        df = pd.read_csv(file_path)  # CSV 파일 읽기
        df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')  # 날짜 변환
        srtnCd = df['srtnCd'].iloc[0]  # 종목코드 추출
        
        # 날짜 필터링 (만약 날짜가 지정되었다면)
        if self.start_date:
            df = df[df['basDt'] >= self.start_date]
        if self.end_date:
            df = df[df['basDt'] <= self.end_date]
            
        return df.sort_values('basDt'), srtnCd  # 날짜순 정렬 후 반환
    
    def load_news_data_from_pricefile(self, pricefile_path):
        """
        주가 파일을 기반으로 뉴스 데이터 로드, 지정된 날짜 범위만 필터링
        """
        pricefile = os.path.basename(pricefile_path)
        news_filename = self.get_news_basename_from_pricefile(pricefile)
        news_file_path = os.path.join(self.news_dir, news_filename)
        
        if not os.path.exists(news_file_path):
            print(f"뉴스 파일 없음: {news_filename}")
            return pd.DataFrame(columns=['basDt', 'news_text'])
        
        df_news = pd.read_csv(news_file_path, header=0)
        df_news['basDt'] = pd.to_datetime(df_news['날짜'], format='%Y%m%d')
        df_news['news_text'] = df_news['제목']
        
        # 날짜 필터링 (만약 날짜가 지정되었다면)
        if self.start_date:
            df_news = df_news[df_news['basDt'] >= self.start_date]
        if self.end_date:
            df_news = df_news[df_news['basDt'] <= self.end_date]
            
        return df_news.sort_values('basDt')[['basDt', 'news_text']]
    
    def prepare_data(self, price_df, news_df):
        """
        모델링을 위한 데이터 준비
        """
        # 필수 컬럼이 price_df에 있는지 확인
        required_cols = ['basDt', 'clpr']
        for col in required_cols:
            if col not in price_df.columns:
                raise ValueError(f"필수 컬럼 {col}이 price_df에 없습니다")
        
        # 모델링을 위한 특성 선택
        features = ['clpr']  # 간단하게 종가만 사용
        if 'trqu' in price_df.columns:
            features.append('trqu')  # 거래량
        if 'mkp' in price_df.columns:
            features.append('mkp')   # 시장 가격
        
        # 특성 행렬 생성
        X = price_df[features].values
        
        # 데이터 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 날짜와 뉴스 텍스트를 매핑하는 딕셔너리 생성
        news_dict = {}
        if not news_df.empty:
            for _, row in news_df.iterrows():
                date = row['basDt']
                text = row['news_text']
                if date in news_dict:
                    news_dict[date] += " " + text
                else:
                    news_dict[date] = text
        
        return X_scaled, news_dict, price_df['basDt'].values, scaler
    
    def collate_fn(self, batch):
        """
        DataLoader를 위한 커스텀 콜레이트 함수
        """
        stock_seqs, stock_targets, news_texts = zip(*batch)
        
        # 텐서 스택킹
        stock_seqs = torch.stack(stock_seqs)
        stock_targets = torch.stack(stock_targets)
        
        return stock_seqs, stock_targets, list(news_texts)
    
    def train_model(self, X_scaled, news_dict, dates, srtnCd):
        """
        융합 모델 학습
        """
        # 데이터셋 생성
        dataset = StockNewsDataset(X_scaled, news_dict, dates, self.sequence_length, self.prediction_length)
        
        # 데이터셋이 비어있는지 확인
        if len(dataset) == 0:
            print(f"{srtnCd}에 대한 데이터셋이 비어있습니다. 건너뜁니다.")
            return None
        
        dataloader = DataLoader(
            dataset, 
            batch_size=min(self.batch_size, len(dataset)), 
            shuffle=True, 
            collate_fn=self.collate_fn
        )
        
        # 모델 정의
        feature_dim = X_scaled.shape[1]
        model = FusionModel(
            seq_len=self.sequence_length,
            pred_len=self.prediction_length,
            enc_in=feature_dim,
            kernel_size=25
        )
        model.to(self.device)
        
        # 손실 함수 및 옵티마이저 정의
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # 학습 루프
        best_loss = float('inf')
        best_model_state = None
        model.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            for batch_idx, (stock_seqs, stock_targets, news_texts) in enumerate(dataloader):
                stock_seqs = stock_seqs.to(self.device)
                stock_targets = stock_targets.to(self.device)
                
                # 순전파
                predictions, _ = model(stock_seqs, news_texts)
                loss = criterion(predictions, stock_targets)
                
                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
            
            # 최고 모델 저장
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict().copy()
        
        # 최고 모델 로드
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model
    
    def save_model(self, model, scaler, srtnCd):
        """
        모델 및 관련 데이터 저장
        """

        padded_srtnCd = str(srtnCd).zfill(6)
        stock_dir = os.path.join(self.output_dir, f"N{padded_srtnCd}")
        os.makedirs(stock_dir, exist_ok=True)
        
        # 모델 저장
        torch.save(model.state_dict(), os.path.join(stock_dir, 'dlinear_model.h5'))
        
        # 스케일러 저장
        joblib.dump(scaler, os.path.join(stock_dir, 'dlinear_scaler.joblib'))
        
        # 기간 정보 저장
        with open(os.path.join(stock_dir, 'period_info.txt'), 'w', encoding='utf-8') as f:
            f.write(f"sequence_length={self.sequence_length}\nperiod=5")
    
    def process_file(self, file_path):
        """
        단일 주가 파일 처리
        """
        print(f"파일 처리 중: {file_path}")
        
        try:
            # 데이터 로드
            price_df, srtnCd = self.load_price_data(file_path)
            news_df = self.load_news_data_from_pricefile(file_path)
            
            # 충분한 데이터가 있는지 확인
            if len(price_df) < (self.sequence_length + self.prediction_length):
                print(f"{file_path}에 충분한 데이터가 없습니다. 건너뜁니다.")
                return None
            
            # 데이터 준비
            X_scaled, news_dict, dates, scaler = self.prepare_data(price_df, news_df)
            
            # 모델 학습
            model = self.train_model(X_scaled, news_dict, dates, srtnCd)
            
            if model is not None:
                # 모델 저장
                self.save_model(model, scaler, srtnCd)
                print(f"{srtnCd}에 대한 모델 학습 및 저장 완료")
                return model
            else:
                print(f"{srtnCd}에 대한 모델 학습 실패")
                return None
                
        except Exception as e:
            print(f"{file_path} 처리 중 오류 발생: {str(e)}")
            return None
    
    def run(self):
        """
        모든 주가 파일 처리
        """
        price_files = [f for f in os.listdir(self.base_dir) if f.endswith('_KOSPI.csv')]
        
        for file in price_files:
            file_path = os.path.join(self.base_dir, file)
            self.process_file(file_path)
            print(f"처리 완료: {file}")

def validate_date_format(date_str):
    """
    날짜 형식이 올바른지 검증하는 함수
    """
    if not date_str:  # 빈 문자열이면 None 반환
        return None
    
    try:
        # YYYYMMDD 형식인지 확인
        if len(date_str) != 8 or not date_str.isdigit():
            return None
        
        # 날짜 객체로 변환 가능한지 확인
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:])
        
        datetime.datetime(year, month, day)
        
        # 문제 없으면 datetime 객체 반환
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None

def get_date_input(prompt):
    """
    사용자로부터 날짜 입력을 받는 함수
    """
    while True:
        date_str = input(prompt)
        
        # 빈 입력은 None으로 처리
        if not date_str.strip():
            return None
        
        # 날짜 형식 검증
        date_obj = validate_date_format(date_str)
        if date_obj:
            return date_obj
        else:
            print("잘못된 날짜 형식입니다. YYYYMMDD 형식으로 다시 입력해주세요. (예: 20200101)")

def main():
    """
    로보 어드바이저 실행을 위한 메인 함수
    """
    print("=" * 50)
    print("주가 예측 로보 어드바이저")
    print("=" * 50)
    print("분석할 날짜 범위를 지정해주세요. (YYYYMMDD 형식, 입력 없으면 모든 데이터 사용)")
    
    # 날짜 입력 받기
    start_date = get_date_input("시작 날짜 (YYYYMMDD): ")
    end_date = get_date_input("종료 날짜 (YYYYMMDD): ")
    
    # 날짜 범위 출력
    if start_date and end_date:
        print(f"분석 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    elif start_date:
        print(f"분석 기간: {start_date.strftime('%Y-%m-%d')} 이후")
    elif end_date:
        print(f"분석 기간: {end_date.strftime('%Y-%m-%d')} 이전")
    else:
        print("분석 기간: 전체 기간")
    
    # 로보 어드바이저 실행
    advisor = RoboAdvisor(
        sequence_length=60,  # 60일의 과거 데이터
        prediction_length=5,  # 5일 후를 예측
        batch_size=32,
        epochs=20,  # CPU 학습을 위해 에포크 수 조정
        learning_rate=0.001,
        start_date=start_date,
        end_date=end_date
    )
    
    advisor.run()

if __name__ == "__main__":
    main()
