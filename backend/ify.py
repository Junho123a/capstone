from flask import Flask, jsonify, render_template, request
import requests
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
import yfinance as yf
from newspaper import Article
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

import os
import io
import base64
from flask_socketio import SocketIO
import time

# Flask 앱 초기화
app = Flask(__name__)
socketio = SocketIO(app, logger=True, engineio_logger=True)

# NLTK 리소스 다운로드
nltk.download('vader_lexicon', quiet=True)

# 필요한 디렉토리 생성
os.makedirs('static', exist_ok=True)
os.makedirs('data', exist_ok=True)

def fetch_news_from_api(company_name, days=30):
    """NewsAPI를 사용하여 특정 회사 관련 영어 뉴스 수집"""
    API_KEY = 'f4aa6d7355cf4d66a41d3bff4b3208bc'
    BASE_URL = 'https://newsapi.org/v2/everything'
    
    # 날짜 범위 설정
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    params = {
        'q': company_name,
        'from': start_date,
        'to': end_date,
        'language': 'en',
        'sortBy': 'publishedAt',
        'apiKey': API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        
        news_list = []
        for article in articles:
            news_list.append({
                'title': article['title'],
                'url': article['url'],
                'publishedAt': article['publishedAt'],
                'source': article['source']['name']
            })
        
        return pd.DataFrame(news_list)
    else:
        print(f"API 요청 실패: {response.status_code}")
        return pd.DataFrame()

def extract_article_content(url):
    """Newspaper3k를 사용하여 기사 내용 추출"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"기사 추출 실패: {e}")
        return ""

def analyze_sentiment_en(text):
    """영어 텍스트 감성 분석"""
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

def get_stock_data(ticker, start_date, end_date):
    """Yahoo Finance에서 주가 데이터 가져오기"""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def create_sequences(data, target, seq_length):
    """LSTM 입력용 시퀀스 데이터 생성"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

def build_model(input_shape):
    """LSTM 모델 구축"""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

class TrainingCallback:
    def __init__(self, socketio):
        self.socketio = socketio
        self.epoch = 0
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        logs = logs or {}
        
        # 진행 상황 업데이트
        progress = int((epoch + 1) / self.params['epochs'] * 100)
        self.socketio.emit('training_progress', {'progress': progress}, broadcast=True)
        
        # 학습 지표 업데이트
        metrics_data = {
            'epoch': epoch + 1,
            'loss': logs.get('loss', 0),
            'val_loss': logs.get('val_loss', 0),
            'accuracy': logs.get('accuracy', 0),
            'val_accuracy': logs.get('val_accuracy', 0)
        }
        self.socketio.emit('training_metrics', metrics_data, broadcast=True)
        
        # 손실 그래프 업데이트
        self.train_loss.append(logs.get('loss', 0))
        self.val_loss.append(logs.get('val_loss', 0))
        self._update_loss_chart()
        
        # 정확도 그래프 업데이트
        self.train_acc.append(logs.get('accuracy', 0))
        self.val_acc.append(logs.get('val_accuracy', 0))
        self._update_accuracy_chart()
        
        # 소켓 이벤트가 클라이언트에 전달되도록 잠시 대기
        time.sleep(0.1)
    
    def _update_loss_chart(self):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_loss) + 1)
        plt.plot(epochs, self.train_loss, 'b-', label='학습 손실')
        plt.plot(epochs, self.val_loss, 'r-', label='검증 손실')
        plt.title('학습 및 검증 손실')
        plt.xlabel('에포크')
        plt.ylabel('손실')
        plt.legend()
        plt.grid(True)
        
        # 이미지를 base64로 인코딩
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        self.socketio.emit('loss_chart', {'chart': f'data:image/png;base64,{plot_url}'}, broadcast=True)
    
    def _update_accuracy_chart(self):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_acc) + 1)
        plt.plot(epochs, self.train_acc, 'b-', label='학습 정확도')
        plt.plot(epochs, self.val_acc, 'r-', label='검증 정확도')
        plt.title('학습 및 검증 정확도')
        plt.xlabel('에포크')
        plt.ylabel('정확도')
        plt.legend()
        plt.grid(True)
        
        # 이미지를 base64로 인코딩
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        self.socketio.emit('accuracy_chart', {'chart': f'data:image/png;base64,{plot_url}'}, broadcast=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train')
def train_view():
    return render_template('training.html')

@app.route('/get_news/<company>')
def get_news(company):
    """뉴스 API 엔드포인트"""
    news_df = fetch_news_from_api(company)
    
    if not news_df.empty:
        # 기사 내용 추출
        news_df['content'] = news_df['url'].apply(extract_article_content)
        
        # 데이터프레임을 딕셔너리 리스트로 변환
        news_list = news_df.to_dict('records')
        
        # 결과 저장
        news_df.to_csv(f"data/{company}_news.csv", index=False, encoding='utf-8')
        
        return jsonify(news_list)
    else:
        return jsonify({"error": "뉴스를 가져오는데 실패했습니다."})

@app.route('/analyze/<company>')
def analyze_sentiment(company):
    """감성 분석 API 엔드포인트"""
    try:
        # 뉴스 데이터 불러오기
        news_file = f"data/{company}_news.csv"
        
        # 파일이 없으면 뉴스 데이터 가져오기
        if not os.path.exists(news_file):
            news_df = fetch_news_from_api(company)
            news_df['content'] = news_df['url'].apply(extract_article_content)
            news_df.to_csv(news_file, index=False, encoding='utf-8')
        else:
            news_df = pd.read_csv(news_file, encoding='utf-8')
        
        # 감성 분석 수행
        sentiment_results = []
        for _, row in news_df.iterrows():
            content = row['content']
            if not content or pd.isna(content):
                continue
                
            sentiment = analyze_sentiment_en(content)
            sentiment_results.append({
                'date': row['publishedAt'][:10],
                'title': row['title'],
                'pos': sentiment['pos'],
                'neg': sentiment['neg'],
                'neu': sentiment['neu'],
                'compound': sentiment['compound']
            })
        
        sentiment_df = pd.DataFrame(sentiment_results)
        
        # 날짜별로 감성 점수 집계
        daily_sentiment = sentiment_df.groupby('date').agg({
            'pos': 'mean',
            'neg': 'mean',
            'neu': 'mean',
            'compound': 'mean'
        }).reset_index()
        
        daily_sentiment.to_csv(f"data/{company}_sentiment.csv", index=False)
        
        return jsonify(daily_sentiment.to_dict('records'))
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/stock/<ticker>')
def get_stock(ticker):
    """주가 데이터 API 엔드포인트"""
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        stock_data = get_stock_data(ticker, start_date, end_date)
        stock_data = stock_data.reset_index()
        stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')
        
        stock_data.to_csv(f"data/{ticker}_stock.csv", index=False)
        
        return jsonify(stock_data.to_dict('records'))
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict', methods=['POST'])
def predict_stock():
    """주가 예측 API 엔드포인트"""
    try:
        data = request.get_json()
        company = data.get('company', 'AAPL')
        
        # 1. 데이터 준비
        sentiment_file = f"data/{company}_sentiment.csv"
        stock_file = f"data/{company}_stock.csv"
        
        # 파일이 없으면 데이터 가져오기
        if not os.path.exists(sentiment_file):
            return jsonify({"error": "먼저 감성 분석을 수행해주세요."})
        
        if not os.path.exists(stock_file):
            return jsonify({"error": "먼저 주가 데이터를 가져와주세요."})
        
        # 2. 데이터 로드
        stock_data = pd.read_csv(stock_file)
        sentiment_data = pd.read_csv(sentiment_file)
        
        # 3. 데이터 병합 및 전처리
        merged_data = pd.merge(stock_data, sentiment_data, left_on='Date', right_on='date', how='left')
        if 'date' in merged_data.columns:
            merged_data = merged_data.drop('date', axis=1)
        
        merged_data = merged_data.fillna(method='ffill')
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'pos', 'neg', 'neu', 'compound']
        target = 'Close'
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        scaled_features = scaler_X.fit_transform(merged_data[features])
        scaled_target = scaler_y.fit_transform(merged_data[[target]])
        
        # 4. 시퀀스 데이터 생성
        seq_length = 10
        X, y = create_sequences(scaled_features, scaled_target, seq_length)
        
        # 5. 학습/테스트 데이터 분할
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 6. 모델 구축 및 학습
        model = build_model((X_train.shape[1], X_train.shape[2]))
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # 학습 콜백 설정
        callback = TrainingCallback(socketio)
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop, callback],
            verbose=0
        )
        
        # 7. 모델 평가
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        # 8. 미래 주가 예측
        latest_data = scaled_features[-seq_length:]
        X_pred = np.array([latest_data])
        prediction = model.predict(X_pred)
        
        # 예측값 역정규화
        predicted_price = scaler_y.inverse_transform(prediction)[0][0]
        current_price = merged_data['Close'].iloc[-1]
        
        # 9. 학습 곡선 시각화
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('모델 학습 곡선')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('static/learning_curve.png')
        plt.close()
        
        # 10. 예측 결과 시각화
        plt.figure(figsize=(12, 6))
        
        # 테스트 데이터 역정규화
        y_test_actual = scaler_y.inverse_transform(y_test)
        y_pred_actual = scaler_y.inverse_transform(y_pred)
        
        plt.plot(y_test_actual, label='실제 주가')
        plt.plot(y_pred_actual, label='예측 주가')
        plt.title('주가 예측 결과')
        plt.xlabel('시간')
        plt.ylabel('주가')
        plt.legend()
        plt.savefig('static/prediction_result.png')
        plt.close()
        
        # 11. 최근 주가 추세와 예측 시각화
        plt.figure(figsize=(12, 6))
        plt.plot(merged_data['Date'].values[-30:], merged_data['Close'].values[-30:], 'b-o', label='최근 주가')
        plt.axhline(y=predicted_price, color='r', linestyle='--', label=f'예측 주가: {predicted_price:.2f}')
        plt.title('주가 예측')
        plt.xlabel('날짜')
        plt.ylabel('주가')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/future_prediction.png')
        plt.close()
        
        # 12. 결과 반환
        result = {
            'company': company,
            'predicted_price': float(predicted_price),
            'current_price': float(current_price),
            'change_percent': float(((predicted_price - current_price) / current_price * 100)),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'charts': {
                'learning_curve': '/static/learning_curve.png',
                'prediction_result': '/static/prediction_result.png',
                'future_prediction': '/static/future_prediction.png'
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True)
