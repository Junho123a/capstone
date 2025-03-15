import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
company = "AAPL"
stock_data = pd.read_csv("apple_stock.csv")
sentiment_data = pd.read_csv(f"{company}_sentiment.csv")

# 날짜 기준으로 데이터 병합
merged_data = pd.merge(stock_data, sentiment_data, left_on='Date', right_on='date', how='left')
merged_data = merged_data.drop('date', axis=1)

# 결측치 처리
merged_data = merged_data.fillna(method='ffill')  # 앞의 값으로 채우기

# 특성 선택
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'pos', 'neg', 'neu', 'compound']
target = 'Close'  # 종가 예측

# 데이터 정규화
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# 특성 정규화
scaled_features = scaler_X.fit_transform(merged_data[features])
scaled_target = scaler_y.fit_transform(merged_data[[target]])

# LSTM 입력용 시퀀스 데이터 생성
def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

# 시퀀스 길이 설정
seq_length = 10  # 10일 데이터로 다음날 예측 (영어 뉴스 데이터가 더 풍부하므로 시퀀스 길이 증가)

# 시퀀스 데이터 생성
X, y = create_sequences(scaled_features, scaled_target, seq_length)

# 학습/테스트 데이터 분할
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 데이터 저장
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# 스케일러 저장 (나중에 예측값을 원래 스케일로 변환하기 위해)
import pickle
with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print("데이터 전처리가 완료되었습니다.")
