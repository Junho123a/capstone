import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
def load_data(news_csv_path, stock_csv_path):
    """
    뉴스 데이터와 주가 데이터를 로드하는 함수
    
    Args:
        news_csv_path (str): 뉴스 데이터 CSV 파일 경로
        stock_csv_path (str): 주가 데이터 CSV 파일 경로
    
    Returns:
        tuple: (뉴스 데이터프레임, 주가 데이터프레임)
    """
    news_df = pd.read_csv(news_csv_path)
    stock_df = pd.read_csv(stock_csv_path)
    
    # 날짜 형식 통일
    if 'Date' in news_df.columns:
        news_df['Date'] = pd.to_datetime(news_df['Date'])
    
    if 'Date' in stock_df.columns:
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
    return news_df, stock_df

# 일별 뉴스 제목 병합
def aggregate_daily_news(news_df):
    """
    날짜별로 뉴스 제목을 합치는 함수
    
    Args:
        news_df (DataFrame): 뉴스 데이터프레임 (Date, Title 컬럼 필요)
    
    Returns:
        DataFrame: 날짜별로 뉴스 제목이 합쳐진 데이터프레임
    """
    # 하루에 여러 개의 뉴스가 있을 경우, 모든 제목을 하나의 문자열로 합침
    news_df_grouped = news_df.groupby('Date')['Title'].apply(' '.join).reset_index()
    return news_df_grouped

# TF-IDF 벡터화
def vectorize_news(news_df_grouped):
    """
    TF-IDF를 사용하여 뉴스 제목을 벡터화하는 함수
    
    Args:
        news_df_grouped (DataFrame): 날짜별로 뉴스 제목이 합쳐진 데이터프레임
    
    Returns:
        tuple: (TF-IDF 행렬, TF-IDF 벡터라이저)
    """
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(news_df_grouped['Title'])
    return tfidf_matrix, vectorizer

# 주가 방향 추가
def add_price_direction(news_df_grouped, stock_df):
    """
    뉴스 데이터에 주가 방향(상승/하락) 정보를 추가하는 함수
    
    Args:
        news_df_grouped (DataFrame): 날짜별로 뉴스 제목이 합쳐진 데이터프레임
        stock_df (DataFrame): 주가 데이터프레임
    
    Returns:
        DataFrame: 주가 방향 정보가 추가된 뉴스 데이터프레임
    """
    # 전일 대비 주가 변동 방향 계산
    stock_df['PriceChange'] = stock_df['Close'].diff()
    stock_df['Direction'] = stock_df['PriceChange'].apply(lambda x: 1 if x > 0 else 0)
    
    # 주가 데이터를 뉴스 데이터에 병합
    merged_df = pd.merge(news_df_grouped, stock_df[['Date', 'Direction']], on='Date', how='inner')
    
    return merged_df

# 유사한 날짜 찾기
def find_similar_dates(current_news, historical_tfidf_matrix, merged_df, vectorizer, top_n=5):
    """
    현재 뉴스와 유사한 과거 뉴스 날짜를 찾는 함수
    
    Args:
        current_news (str): 현재 뉴스 제목
        historical_tfidf_matrix (sparse matrix): 과거 뉴스의 TF-IDF 행렬
        merged_df (DataFrame): 뉴스와 주가 방향이 병합된 데이터프레임
        vectorizer (TfidfVectorizer): 학습된 TF-IDF 벡터라이저
        top_n (int, optional): 반환할 유사한 날짜의 수. 기본값은 5.
    
    Returns:
        DataFrame: 유사도가 높은 상위 n개 날짜와 그 날의 주가 방향
    """
    # 현재 뉴스를 벡터화
    current_tfidf = vectorizer.transform([current_news])
    
    # 현재 뉴스와 과거 뉴스 간의 코사인 유사도 계산
    cosine_similarities = cosine_similarity(current_tfidf, historical_tfidf_matrix).flatten()
    
    # 유사도와 날짜, 주가 방향을 포함한 데이터프레임 생성
    similar_df = pd.DataFrame({
        'Date': merged_df['Date'],
        'Similarity': cosine_similarities,
        'Direction': merged_df['Direction']
    })
    
    # 유사도 순으로 정렬하여 상위 n개 반환
    top_similar = similar_df.sort_values('Similarity', ascending=False).head(top_n)
    
    return top_similar

# 주가 방향 예측
def predict_price_direction(top_similar, similarity_weight=True):
    """
    유사한 과거 날짜들의 주가 방향을 기반으로 현재 주가 방향을 예측하는 함수
    
    Args:
        top_similar (DataFrame): 유사도가 높은 상위 n개 날짜와 그 날의 주가 방향
        similarity_weight (bool, optional): 유사도를 가중치로 사용할지 여부. 기본값은 True.
    
    Returns:
        tuple: (예측된 주가 방향(0 또는 1), 확률)
    """
    if similarity_weight:
        # 유사도를 가중치로 사용
        weighted_sum = (top_similar['Similarity'] * top_similar['Direction']).sum()
        total_similarity = top_similar['Similarity'].sum()
        probability = weighted_sum / total_similarity
        
        predicted_direction = 1 if probability > 0.5 else 0
        return predicted_direction, probability
    else:
        # 단순 다수결 사용
        direction_count = top_similar['Direction'].value_counts()
        if len(direction_count) > 1:
            predicted_direction = direction_count.idxmax()
            probability = direction_count.max() / len(top_similar)
        else:
            predicted_direction = direction_count.index[0]
            probability = 1.0
        
        return predicted_direction, probability

# 결과 시각화
def visualize_results(top_similar, predicted_direction, probability):
    """
    예측 결과를 시각화하는 함수
    
    Args:
        top_similar (DataFrame): 유사도가 높은 상위 n개 날짜와 그 날의 주가 방향
        predicted_direction (int): 예측된 주가 방향(0 또는 1)
        probability (float): 예측 확률
    """
    plt.figure(figsize=(12, 6))
    
    # 유사한 날짜와 주가 방향 시각화
    plt.subplot(1, 2, 1)
    colors = ['red' if d == 0 else 'green' for d in top_similar['Direction']]
    plt.barh(top_similar['Date'].dt.strftime('%Y-%m-%d'), top_similar['Similarity'], color=colors)
    plt.xlabel('유사도')
    plt.ylabel('날짜')
    plt.title('유사한 과거 날짜와 주가 방향')
    plt.grid(axis='x')
    
    # 예측 결과 시각화
    plt.subplot(1, 2, 2)
    direction_text = '상승' if predicted_direction == 1 else '하락'
    plt.pie([probability, 1-probability], labels=[direction_text, '반대 방향'], 
            autopct='%1.1f%%', colors=['green' if predicted_direction == 1 else 'red', 'lightgrey'])
    plt.title(f'주가 예측 결과: {direction_text} (확률: {probability:.2f})')
    
    plt.tight_layout()
    plt.show()

# 메인 함수
def predict_stock_price(news_csv_path, stock_csv_path, current_news, top_n=5, similarity_weight=True):
    """
    현재 뉴스를 기반으로 주가 방향을 예측하는 메인 함수
    
    Args:
        news_csv_path (str): 과거 뉴스 데이터 CSV 파일 경로
        stock_csv_path (str): 과거 주가 데이터 CSV 파일 경로
        current_news (str): 현재 뉴스 제목
        top_n (int, optional): 참고할 유사한 과거 날짜의 수. 기본값은 5.
        similarity_weight (bool, optional): 유사도를 가중치로 사용할지 여부. 기본값은 True.
    
    Returns:
        tuple: (예측된 주가 방향(0 또는 1), 확률)
    """
    # 데이터 로드
    news_df, stock_df = load_data(news_csv_path, stock_csv_path)
    
    # 일별 뉴스 제목 병합
    news_df_grouped = aggregate_daily_news(news_df)
    
    # 주가 방향 추가
    merged_df = add_price_direction(news_df_grouped, stock_df)
    
    # TF-IDF 벡터화
    historical_tfidf_matrix, vectorizer = vectorize_news(merged_df)
    
    # 유사한 날짜 찾기
    top_similar = find_similar_dates(current_news, historical_tfidf_matrix, merged_df, vectorizer, top_n)
    
    # 주가 방향 예측
    predicted_direction, probability = predict_price_direction(top_similar, similarity_weight)
    
    # 결과 시각화
    visualize_results(top_similar, predicted_direction, probability)
    
    return predicted_direction, probability


# 실행 코드
if __name__ == "__main__":
    # 데이터 경로 설정
    news_csv_path = "news_data.csv"  # 뉴스 제목과 날짜가 포함된 CSV 파일
    stock_csv_path = "stock_data.csv"  # 주가 데이터가 포함된 CSV 파일
    
    # 현재 뉴스 제목 (예측하고자 하는 날짜의 뉴스)
    current_news = "삼성전자, 신형 갤럭시 출시 임박... 시장 기대감 상승"
    
    # 예측 실행
    predicted_direction, probability = predict_stock_price(news_csv_path, stock_csv_path, current_news)
    
    # 결과 출력
    direction_text = '상승' if predicted_direction == 1 else '하락'
    print(f"현재 뉴스: {current_news}")
    print(f"주가 예측 결과: {direction_text} (확률: {probability:.2f})")
