import requests
from newspaper import Article
import pandas as pd
from datetime import datetime, timedelta

def fetch_news_from_api(company_name, days=30):
    """NewsAPI를 사용하여 특정 회사 관련 영어 뉴스 수집"""
    API_KEY = 'f4aa6d7355cf4d66a41d3bff4b3208bc'  # NewsAPI 키로 변경 필요
    BASE_URL = 'https://newsapi.org/v2/everything'
    
    # 날짜 범위 설정
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    params = {
        'q': company_name,
        'from': start_date,
        'to': end_date,
        'language': 'en',  # 영어 기사
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

# 애플 관련 뉴스 수집 예시
company = "Apple"  # 또는 "AAPL"
news_df = fetch_news_from_api(company)

# 기사 내용 추출
news_df['content'] = news_df['url'].apply(extract_article_content)

# 결과 저장
news_df.to_csv(f"{company}_news.csv", index=False, encoding='utf-8')
print(f"{len(news_df)} 개의 기사가 수집되었습니다.")
