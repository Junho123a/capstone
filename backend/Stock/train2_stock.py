import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, save_model
from keras.layers import LSTM, Dropout, Dense

class LSTMTrainingSystem:
    """산업용 LSTM 학습 시스템"""
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = self.build_lstm_model()

    def build_lstm_model(self):
        """프로덕션 레벨 LSTM 아키텍처"""
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def load_data(self, file_path):
        """날짜 정보 제거된 데이터 로드"""
        df = pd.read_csv(file_path)
        df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')
        df.sort_values('basDt', inplace=True)
        return df[['clpr']]  # 날짜 컬럼 제외

    def create_dataset(self, data):
        """시계열 데이터셋 생성"""
        X, y = [], []
        scaled = self.scaler.fit_transform(data)
        for i in range(len(scaled) - self.sequence_length):
            X.append(scaled[i:i+self.sequence_length])
            y.append(scaled[i+self.sequence_length])
        return np.array(X), np.array(y)


    def train_and_save(self, csv_path, save_dir='saved_models'):
        """전체 훈련 프로세스 실행"""
        # 데이터 준비
        df = self.load_data(csv_path)
        X, y = self.create_dataset(df)
        
        # 데이터 분할
        split = int(len(X)*0.8)
        X_train, y_train = X[:split], y[:split]
        
        # 모델 훈련
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
        
        # 시스템 저장
        os.makedirs(save_dir, exist_ok=True)
        save_model(self.model, os.path.join(save_dir, 'lstm_model.h5'))
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.joblib'))
        print(f"시스템이 {save_dir}에 저장되었습니다")

def select_csv_file():
    """CSV 파일 선택 기능"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'CSV'))
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError("CSV 디렉토리에 파일이 없습니다")
    
    print("학습 가능한 CSV 파일 목록:")
    for idx, file in enumerate(csv_files, 1):
        print(f"{idx}. {file}")
    
    while True:
        try:
            choice = int(input("선택할 파일 번호 입력: ")) - 1
            selected = os.path.join(base_dir, csv_files[choice])
            print(f"선택된 파일: {selected}")
            return selected
        except (ValueError, IndexError):
            print("잘못된 입력입니다. 다시 시도하세요.")

if __name__ == "__main__":
    try:
        csv_path = select_csv_file()
        trainer = LSTMTrainingSystem()
        trainer.train_and_save(csv_path)
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        exit(1)
