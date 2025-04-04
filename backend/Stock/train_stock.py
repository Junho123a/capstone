import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Model, save_model
from keras.layers import Input, Dense, Subtract, Add, Reshape, GlobalAveragePooling1D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import joblib

class MultiStockDLinearTrainer:
    def __init__(self, base_dir, output_dir='dlinear_models',
                 target_start=datetime(2020, 1, 2),
                 target_end=datetime(2023, 12, 31),
                 sequence_length=20):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.target_start = target_start
        self.target_end = target_end
        self.sequence_length = sequence_length
        os.makedirs(output_dir, exist_ok=True)

    def _is_valid_file(self, filename):
        """파일명 날짜 유효성 검증"""
        try:
            date_matches = re.findall(r'_(\d{8})_(\d{8})_', filename)
            if not date_matches:
                return False
                
            file_start = datetime.strptime(date_matches[0][0], '%Y%m%d')
            file_end = datetime.strptime(date_matches[0][1], '%Y%m%d')
            return (file_start >= self.target_start) and (file_end <= self.target_end)
        except ValueError:
            return False

    def build_dlinear_model(self):
        """DLinear 아키텍처 구현"""
        input_layer = Input(shape=(self.sequence_length, 1))
        
        # 추세 성분 추출
        trend = GlobalAveragePooling1D()(input_layer)
        trend = Reshape((1,))(trend)
        
        # 계절성 성분 분해
        trend_expanded = Reshape((1, 1))(trend)
        seasonal = Subtract()([input_layer, trend_expanded])
        
        # 계절성 성분 처리
        seasonal_flat = Flatten()(seasonal)
        seasonal_output = Dense(1, use_bias=False)(seasonal_flat)
        
        # 추세 성분 처리
        trend_output = Dense(1, use_bias=False)(trend)
        
        # 최종 출력 결합
        output = Add()([trend_output, seasonal_output])
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(0.001), loss='mse')
        return model

    def load_data(self, file_path):
        """종목 데이터 로드"""
        df = pd.read_csv(file_path)
        df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')
        return df.sort_values('basDt')[['clpr']]

    def create_dataset(self, data, scaler):
        """시계열 데이터셋 생성"""
        scaled = scaler.fit_transform(data)
        X, y = [], []
        for i in range(len(scaled)-self.sequence_length):
            X.append(scaled[i:i+self.sequence_length])
            y.append(scaled[i+self.sequence_length])
        return np.array(X), np.array(y)

    def train_single_stock(self, csv_path, stock_name):
        """단일 종목 학습 프로세스"""
        print(f"\n▶ {stock_name} 학습 시작")
        
        # 데이터 검증
        df = self.load_data(csv_path)
        if len(df) < self.sequence_length + 1:
            print(f"  ⚠ {stock_name} 데이터 부족: {len(df)} < {self.sequence_length+1}")
            return False
            
        # 데이터 준비
        scaler = MinMaxScaler()
        X, y = self.create_dataset(df, scaler)
        
        # 모델 초기화
        model = self.build_dlinear_model()
        
        # 시계열 교차 검증
        tscv = TimeSeriesSplit(n_splits=5)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"  Fold {fold+1}/5 진행중")
            model.fit(X[train_idx], y[train_idx],
                      validation_data=(X[val_idx], y[val_idx]),
                      epochs=100, batch_size=64,
                      callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                      verbose=0)
        
        # 모델 저장
        stock_dir = os.path.join(self.output_dir, stock_name)
        os.makedirs(stock_dir, exist_ok=True)
        save_model(model, os.path.join(stock_dir, 'dlinear_model.h5'))
        joblib.dump(scaler, os.path.join(stock_dir, 'dlinear_scaler.joblib'))
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
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'CSV'))
    
    trainer = MultiStockDLinearTrainer(
        base_dir=base_dir,
        output_dir='dlinear_models',
        target_start=datetime(2020, 1, 2),
        target_end=datetime(2023, 12, 31),
        sequence_length=20
    )
    
    try:
        trainer.train_all_stocks()
    except Exception as e:
        print(f"\n※ 치명적 오류: {str(e)} ※")
