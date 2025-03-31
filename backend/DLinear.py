import numpy as np
import matplotlib.pyplot as plt

class DLinear:
    def __init__(self, input_size, output_size, moving_avg_window=25):
        self.input_size = input_size
        self.output_size = output_size
        self.moving_avg_window = moving_avg_window
        
        # 가중치 초기화
        self.seasonal_weights = np.random.normal(0, 0.01, (input_size, output_size))
        self.seasonal_bias = np.zeros(output_size)
        self.trend_weights = np.random.normal(0, 0.01, (input_size, output_size))
        self.trend_bias = np.zeros(output_size)
    
    # 앞서 정의한 decompose, _moving_average, forward, train, predict 메서드 포함
    
# 데이터 생성 (예: 계절성과 추세가 있는 시계열)
def generate_time_series(n=1000):
    time = np.arange(n)
    # 추세 성분
    trend = 0.01 * time
    # 계절성 성분
    seasonality = 5 * np.sin(2 * np.pi * time / 50)
    # 노이즈
    noise = np.random.normal(0, 1, n)
    # 시계열 데이터
    series = trend + seasonality + noise
    return series.reshape(-1, 1)

# 시계열 데이터 생성
data = generate_time_series(1000)

# 학습 및 테스트 데이터 분할
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# 입력/출력 시퀀스 생성
def create_sequences(data, input_size, output_size):
    X, y = [], []
    for i in range(len(data) - input_size - output_size + 1):
        X.append(data[i:i+input_size])
        y.append(data[i+input_size:i+input_size+output_size])
    return np.array(X), np.array(y)

input_size = 50
output_size = 10
X_train, y_train = create_sequences(train_data, input_size, output_size)
X_test, y_test = create_sequences(test_data, input_size, output_size)

# DLinear 모델 초기화 및 학습
model = DLinear(input_size, output_size)
losses = model.train(X_train, y_train, learning_rate=0.01, epochs=1000)

# 예측
predictions = model.predict(X_test)

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test[0].flatten(), label='Actual')
plt.plot(predictions[0].flatten(), label='Predicted')
plt.title('DLinear Prediction vs Actual')
plt.legend()
plt.show()
