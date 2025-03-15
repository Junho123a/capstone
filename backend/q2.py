import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

# NLTK 리소스 다운로드
nltk.download('vader_lexicon')

def analyze_sentiment_en(text):
    """영어 텍스트 감성 분석"""
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

# 뉴스 데이터 불러오기
company = "AAPL"  # 애플 주식 예시 (영어 기사를 분석할 회사 티커)
news_df = pd.read_csv(f"{company}_news.csv", encoding='utf-8-sig')

# 감성 분석 수행
sentiment_results = []
for _, row in news_df.iterrows():
    content = row['content']
    if not content or pd.isna(content):
        continue
        
    # 영어 감성 분석
    sentiment = analyze_sentiment_en(content)
    
    sentiment_results.append({
        'date': row['publishedAt'][:10],  # 날짜만 추출
        'title': row['title'],
        'pos': sentiment['pos'],
        'neg': sentiment['neg'],
        'neu': sentiment['neu'],
        'compound': sentiment['compound']
    })

# 감성 분석 결과를 데이터프레임으로 변환
sentiment_df = pd.DataFrame(sentiment_results)

# 날짜별로 감성 점수 집계
daily_sentiment = sentiment_df.groupby('date').agg({
    'pos': 'mean',
    'neg': 'mean',
    'neu': 'mean',
    'compound': 'mean'
}).reset_index()

daily_sentiment.to_csv(f"{company}_sentiment.csv", index=False, encoding='utf-8-sig')
print("감성 분석이 완료되었습니다.")
