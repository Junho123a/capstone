import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Model, save_model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import joblib

class MultiStockCNNTrainer:
    def __init__(self, base_dir, output_dir='models', 
                 target_start=datetime(2020, 1, 2), 
                 target_end=datetime(2023, 12, 31)):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.target_start = target_start
        self.target_end = target_end
        self.sequence_length = 20
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

    def build_cnn_model(self):
        """CNN 모델 생성"""
        input_layer = Input(shape=(self.sequence_length, 1))
        x = Conv1D(64, 3, activation='relu')(input_layer)
        x = MaxPooling1D(2)(x)
        x = Conv1D(32, 3, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Flatten()(x)
        x = Dense(50, activation='relu')(x)
        output = Dense(1)(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(0.001), loss='mse')
        return model

    def load_data(self, file_path):
        """CSV 데이터 로드 및 전처리"""
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
        """단일 종목 학습 파이프라인"""
        print(f"\n▶ {stock_name} 학습 시작")
        df = self.load_data(csv_path)
        scaler = MinMaxScaler()
        X, y = self.create_dataset(df, scaler)
        
        model = self.build_cnn_model()
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
        save_model(model, os.path.join(stock_dir, 'cnn_model.h5'))
        joblib.dump(scaler, os.path.join(stock_dir, 'cnn_scaler.joblib'))
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
        trainer = MultiStockCNNTrainer(
            base_dir=base_dir,
            output_dir='saved_models',
            target_start=target_start,
            target_end=target_end
        )
        
        trainer.train_all_stocks()
        
    except Exception as e:
        print(f"\n※ 치명적 오류 발생: {str(e)} ※")
