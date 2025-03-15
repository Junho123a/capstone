import torch
import torch.nn as nn
import torch.optim as optim

class DLinear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DLinear, self).__init__()
        # 선형 레이어
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 순전파
        x = torch.relu(self.fc1(x))  # 첫 번째 선형 레이어 + ReLU 활성화 함수
        x = self.fc2(x)  # 두 번째 선형 레이어
        return x

# 데이터 준비 (예시)
X_train = torch.randn(100, 10)  # 100개의 샘플, 10개의 특성
y_train = torch.randn(100, 1)   # 100개의 목표 값

# 모델, 손실 함수, 최적화 함수 설정
model = DLinear(input_size=10, hidden_size=50, output_size=1)
criterion = nn.MSELoss()  # 평균 제곱 오차 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 훈련 과정
num_epochs = 100
for epoch in range(num_epochs):
    model.train()

    # 순전파
    output = model(X_train)

    # 손실 계산
    loss = criterion(output, y_train)

    # 기울기 초기화
    optimizer.zero_grad()

    # 역전파
    loss.backward()

    # 최적화
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 예측
model.eval()
with torch.no_grad():
    predictions = model(X_train)
    print(predictions)
