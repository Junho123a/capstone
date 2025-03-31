import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model, save_model
from keras.layers import Input, Conv2D, Dense, Reshape, Concatenate, GlobalAvgPool2D
from scipy.fftpack import fft

class TimesNetTrainingSystem:
    def __init__(self, sequence_length=30, period=None):
        self.sequence_length = sequence_length
        self.period = period
        self.scaler = MinMaxScaler()
        self.model = None

    def build_timesnet_model(self):
        input_layer = Input(shape=(self.sequence_length, 1))
        
        if self.sequence_length % self.period != 0:
            raise ValueError(f"Sequence length {self.sequence_length} must be divisible by period {self.period}")
            
        num_periods = self.sequence_length // self.period
        
        x = Reshape((num_periods, self.period, 1))(input_layer)
        
        branches = []
        for k in [3, 5, 7]:
            branch = Conv2D(64, (1, k), padding='same', activation='relu')(x)
            branches.append(branch)
        
        res = Conv2D(64, (1, 1), padding='same')(x)
        branches.append(res)
        
        x = Concatenate(axis=-1)(branches)
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = GlobalAvgPool2D()(x)
        
        output = Dense(1)(x)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')
        df.sort_values('basDt', inplace=True)
        return df[['basDt', 'clpr']]

    def create_dataset(self, data):
        scaled = self.scaler.fit_transform(data[['clpr']])
        
        if len(scaled) < self.sequence_length:
            raise ValueError(f"Data length {len(scaled)} is shorter than sequence length {self.sequence_length}")
            
        X, y = [], []
        for i in range(len(scaled) - self.sequence_length):
            seq = scaled[i:i+self.sequence_length]
            X.append(seq)
            y.append(scaled[i+self.sequence_length])
            
        return np.array(X), np.array(y)

    def find_dominant_period(self, data, max_period=20):
        fft_vals = np.abs(fft(data['clpr'].values))
        freqs = np.fft.fftfreq(len(data))
        
        dominant_freq_idx = np.argmax(fft_vals[1:]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        
        raw_period = int(1/dominant_freq) if dominant_freq != 0 else max_period
        
        divisors = [i for i in range(1, self.sequence_length+1) 
                    if self.sequence_length % i == 0]
        
        return min(divisors, key=lambda x: abs(x - raw_period))

    def train_and_save(self, csv_path, save_dir='saved_models'):
        df = self.load_data(csv_path)
        
        if self.period is None:
            self.period = self.find_dominant_period(df)
        
        print(f"사용할 주기: {self.period}")
        
        self.model = self.build_timesnet_model()
        
        X, y = self.create_dataset(df)
        
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
        
        os.makedirs(save_dir, exist_ok=True)
        
        save_model(self.model, os.path.join(save_dir, 'timesnet_model.h5'))
        
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
        
        trainer = TimesNetTrainingSystem(sequence_length=60)  # 시퀀스 길이를 60으로 조정
        
        trainer.train_and_save(csv_path)

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        exit(1)
