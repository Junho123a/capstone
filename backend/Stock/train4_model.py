import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Model, save_model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import joblib

class CNNTrainingSystem:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = self.build_cnn_model()

    def build_cnn_model(self):
        input_layer = Input(shape=(self.sequence_length, 1))
        
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(50, activation='relu')(x)
        output = Dense(1)(x)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')
        df.sort_values('basDt', inplace=True)
        return df[['clpr']]

    def create_dataset(self, data):
        X, y = [], []
        scaled = self.scaler.fit_transform(data)
        for i in range(len(scaled) - self.sequence_length):
            X.append(scaled[i:i+self.sequence_length])
            y.append(scaled[i+self.sequence_length])
        return np.array(X), np.array(y)

    def train_and_save(self, csv_path, save_dir='saved_models'):
        df = self.load_data(csv_path)
        X, y = self.create_dataset(df)
        
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            self.model.fit(X_train, y_train, 
                           validation_data=(X_val, y_val),
                           epochs=100, 
                           batch_size=64, 
                           callbacks=[early_stopping],
                           verbose=1)
        
        os.makedirs(save_dir, exist_ok=True)
        save_model(self.model, os.path.join(save_dir, 'cnn_model.h5'))
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.joblib'))

        print(f"시스템이 {save_dir}에 저장되었습니다")

def select_csv_file():
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
        trainer = CNNTrainingSystem()
        trainer.train_and_save(csv_path)
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        exit(1)
