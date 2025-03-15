import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 모델 및 스케일러 로드
model = load_model('stock_prediction_model.h5')

# 새로운 데이터 준비 (최근 5일 데이터)
# 실제 구현에서는 최신 데이터를 가져와야 함
recent_data = pd.read_csv("recent_merged_data.csv")  # 최근 데이터 (주가 + 감성 분석)

# 특성 선택
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'pos', 'neg', 'neu', 'compound']

# 데이터 정규화 (학습에 사용한 것과 동일한 스케일러 사용)
scaler_X = MinMaxScaler()
scaler_X.fit(recent_data[features])
scaled_features = scaler_X.transform(recent_data[features])

# 예측용 시퀀스 생성 (최근 5일 데이터)
seq_length = 5
X_pred = np.array([scaled_features[-seq_length:]])

# 예측 수행
prediction = model.predict(X_pred)

# 예측값 역정규화 (실제 주가 스케일로 변환)
scaler_y = MinMaxScaler()
scaler_y.fit(recent_data[['Close']])
predicted_price = scaler_y.inverse_transform(prediction)[0][0]

print(f"내일 예상 주가: {predicted_price:.2f}")

# 최근 주가 추세와 예측 시각화
plt.figure(figsize=(12, 6))
plt.plot(recent_data['Date'].values[-10:], recent_data['Close'].values[-10:], 'b-o', label='최근 주가')
plt.plot(pd.to_datetime(recent_data['Date'].values[-1]) + pd.Timedelta(days=1), 
         predicted_price, 'r-o', label='예측 주가')
plt.title('주가 예측')
plt.xlabel('날짜')
plt.ylabel('주가')
plt.legend()
plt.grid(True)
plt.savefig('future_prediction.png')
plt.close()

print("예측이 완료되었습니다.")
