import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Model, save_model
from keras.layers import Input, Dense, Conv2D, Reshape, Concatenate, GlobalAvgPool2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from scipy.fftpack import fft
import joblib

class MultiStockTimesNetTrainer:
    def __init__(self, base_dir, output_dir='models', 
                 target_start=datetime(2020, 1, 2), 
                 target_end=datetime(2023, 12, 31),
                 sequence_length=60, max_period=20):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.target_start = target_start
        self.target_end = target_end
        self.sequence_length = sequence_length
        self.max_period = max_period
        os.makedirs(output_dir, exist_ok=True)

    def _is_valid_file(self, filename):
        """파일명에서 날짜 추출 후 기간 유효성 검증"""
        try:
            # 파일명에서 날짜 추출 (형식: _YYYYMMDD_YYYYMMDD_)
            date_matches = re.findall(r'_(\d{8})_(\d{8})_', filename)
            if not date_matches:
                return False
                
            file_start = datetime.strptime(date_matches[0][0], '%Y%m%d')
            file_end = datetime.strptime(date_matches[0][1], '%Y%m%d')
            
            # 대상 기간 완전 포함 여부 확인
            return (file_start >= self.target_start) and (file_end <= self.target_end)
            
        except ValueError:
            return False

    def find_dominant_period(self, data, max_period=None):
        """FFT를 사용하여 시계열 데이터의 주요 주기 탐지"""
        if max_period is None:
            max_period = self.max_period
            
        close_prices = data['clpr'].values
        fft_vals = np.abs(fft(close_prices))
        freqs = np.fft.fftfreq(len(close_prices))
        
        # 0이 아닌 주파수 중 가장 강한 신호 탐지
        dominant_freq_idx = np.argmax(fft_vals[1:]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        
        # 주파수를 주기로 변환
        raw_period = int(1/dominant_freq) if dominant_freq != 0 else max_period
        
        # 시퀀스 길이로 나누어 떨어지는 가장 가까운 주기 찾기
        divisors = [i for i in range(1, self.sequence_length+1) 
                   if self.sequence_length % i == 0]
        
        return min(divisors, key=lambda x: abs(x - raw_period))

    def build_timesnet_model(self, period):
        """TimesNet 모델 생성"""
        input_layer = Input(shape=(self.sequence_length, 1))
        
        # 시퀀스를 주기별로 재구성
        num_periods = self.sequence_length // period
        x = Reshape((num_periods, period, 1))(input_layer)
        
        # 다양한 커널 크기의 브랜치 생성
        branches = []
        for k in [3, 5, 7]:
            branch = Conv2D(64, (1, k), padding='same', activation='relu')(x)
            branches.append(branch)
        
        # 잔차 연결 추가
        res = Conv2D(64, (1, 1), padding='same')(x)
        branches.append(res)
        
        # 브랜치 합치기
        x = Concatenate(axis=-1)(branches)
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = GlobalAvgPool2D()(x)
        
        output = Dense(1)(x)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(0.001), loss='mse')
        return model

    def load_data(self, file_path):
        """CSV 데이터 로드 및 전처리"""
        df = pd.read_csv(file_path)
        df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')
        return df.sort_values('basDt')[['basDt', 'clpr']]

    def create_dataset(self, data, scaler):
        """시계열 데이터셋 생성"""
        scaled = scaler.fit_transform(data[['clpr']])
        
        if len(scaled) < self.sequence_length + 1:
            raise ValueError(f"데이터 길이({len(scaled)})가 시퀀스 길이({self.sequence_length})보다 작습니다")
            
        X, y = [], []
        for i in range(len(scaled) - self.sequence_length):
            X.append(scaled[i:i+self.sequence_length])
            y.append(scaled[i+self.sequence_length])
            
        return np.array(X), np.array(y)

    def train_single_stock(self, csv_path, stock_name):
        """단일 종목 학습 파이프라인"""
        print(f"\n▶ {stock_name} 학습 시작")
        
        # 데이터 로드
        df = self.load_data(csv_path)
        if len(df) < self.sequence_length + 1:
            print(f"  ⚠ {stock_name}의 데이터가 부족하여 건너뜁니다 ({len(df)} < {self.sequence_length+1})")
            return False
            
        # 주요 주기 탐지
        period = self.find_dominant_period(df)
        print(f"  - 탐지된 주요 주기: {period}")
        
        # 데이터셋 생성
        scaler = MinMaxScaler()
        X, y = self.create_dataset(df, scaler)
        
        # 모델 생성
        model = self.build_timesnet_model(period)
        
        # 교차 검증
        tscv = TimeSeriesSplit(n_splits=5)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"  Fold {fold+1}/5 진행중")
            model.fit(X[train_idx], y[train_idx],
                      validation_data=(X[val_idx], y[val_idx]),
                      epochs=100, batch_size=64,
                      callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                      verbose=0)
        
        # 종목별 저장 디렉토리 생성
        stock_dir = os.path.join(self.output_dir, stock_name)
        os.makedirs(stock_dir, exist_ok=True)
        
        # 모델 저장
        save_model(model, os.path.join(stock_dir, 'timesnet_model.h5'))
        joblib.dump(scaler, os.path.join(stock_dir, 'timesnet_scaler.joblib'))
        
        # 주기 정보 저장
        with open(os.path.join(stock_dir, 'period_info.txt'), 'w') as f:
            f.write(f"sequence_length={self.sequence_length}\nperiod={period}")
        
        print(f"✔ {stock_name} 학습 완료 | 저장 위치: {stock_dir}")
        return True

    def train_all_stocks(self):
        """전체 학습 프로세스 실행"""
        all_files = os.listdir(self.base_dir)
        valid_files = [f for f in all_files if self._is_valid_file(f)]
        
        if not valid_files:
            raise FileNotFoundError("조건에 맞는 파일이 없습니다")
            
        print(f"※ {len(valid_files)}개의 유효한 파일 발견 ※")
        
        success_count = 0
        for idx, file in enumerate(valid_files, 1):
            # 파일명에서 종목 코드 추출 (첫 번째 언더스코어 기준)
            base_name = os.path.splitext(file)[0]  # 확장자 제거
            stock_name = base_name.split('_')[0]    # 첫 번째 부분 추출
            
            csv_path = os.path.join(self.base_dir, file)
            print(f"\n{'='*40}")
            print(f"[{idx}/{len(valid_files)}] {stock_name} 처리 시작")
            try:
                if self.train_single_stock(csv_path, stock_name):
                    success_count += 1
            except Exception as e:
                print(f"※ {stock_name} 처리 실패: {str(e)}")
                
        print(f"\n학습 완료: {success_count}/{len(valid_files)} 종목 처리됨")


if __name__ == "__main__":
    # CSV 폴더 경로 설정 (파일 위치를 기준으로 상대 경로 설정)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'CSV'))
    
    # 학습 대상 기간 설정
    target_start = datetime(2020, 1, 2)
    target_end = datetime(2023, 12, 31)

    # 학습 시스템 실행
    try:
        trainer = MultiStockTimesNetTrainer(
            base_dir=base_dir,
            output_dir='saved_models',
            target_start=target_start,
            target_end=target_end,
            sequence_length=60  # 시퀀스 길이를 60으로 설정
        )
        
        trainer.train_all_stocks()
        
    except Exception as e:
        print(f"\n※ 치명적 오류 발생: {str(e)} ※")
