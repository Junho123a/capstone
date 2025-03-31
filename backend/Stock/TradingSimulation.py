import urllib.request
import json
import pandas as pd
import datetime
import re

# 클라이언트 정보 설정
client_id = "mhcrCRNNiivdprflWN48"  # 발급받은 클라이언트 아이디 입력
client_secret = "U9z3Kvm6Kf"  # 발급받은 클라이언트 시크릿 입력

# 검색어 설정
search_keyword = input("검색어를 입력하세요: ")
encText = urllib.parse.quote(search_keyword)

# API 요청 URL 생성
url = "https://openapi.naver.com/v1/search/news.json"
url += "?query=" + encText
url += "&display=100"  # 한 번에 가져올 뉴스 개수 (최대 100)
url += "&start=1"  # 검색 시작 위치
url += "&sort=date"  # 정렬 방식 (date: 날짜순, sim: 관련도순)

# API 요청 객체 생성
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id", client_id)
request.add_header("X-Naver-Client-Secret", client_secret)

# API 요청 및 응답 처리
response = urllib.request.urlopen(request)
rescode = response.getcode()

if rescode == 200:
    # 응답 데이터 파싱
    response_body = response.read()
    json_data = json.loads(response_body.decode('utf-8'))
    
    # 결과 출력
    print(f"총 검색 결과: {json_data['total']} 건")
    
    # 데이터프레임으로 변환
    news_df = pd.DataFrame(json_data['items'])
    
    # HTML 태그 제거
    news_df['title'] = news_df['title'].apply(lambda x: re.sub('<.*?>', '', x))
    news_df['description'] = news_df['description'].apply(lambda x: re.sub('<.*?>', '', x))
    
    # pubDate 필드에서 날짜 추출 및 필터링
    news_df['pubDate'] = pd.to_datetime(news_df['pubDate'], format='%a, %d %b %Y %H:%M:%S %z', errors='coerce')

    # 시간대 제거 (tz-naive로 변환)
    news_df['pubDate'] = news_df['pubDate'].dt.tz_localize(None)

    # 2020년대 뉴스 필터링 (2020-01-01 ~ 2029-12-31)
    start_date = "2020-01-01"
    end_date = "2029-12-31"
    filtered_news_df = news_df[(news_df['pubDate'] >= pd.to_datetime(start_date)) & 
                               (news_df['pubDate'] <= pd.to_datetime(end_date))]
    
    # 결과 저장
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"naver_news_{search_keyword}_2020s_{now}.csv"
    filtered_news_df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"2020년대 뉴스가 {filename}에 저장되었습니다.")
else:
    print(f"Error Code: {rescode}")
