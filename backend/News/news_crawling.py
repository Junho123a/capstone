import requests
from bs4 import BeautifulSoup
import time
import datetime
import pandas as pd

# 사용자로부터 검색어와 날짜 범위 입력받기
keyword = input("검색어를 입력하세요: ")
start_date_str = input("시작일을 입력하세요 (예: 2023.01.01): ")
end_date_str = input("종료일을 입력하세요 (예: 2023.12.31): ")
max_pages_per_period = int(input("한 기간당 최대 몇 페이지까지 크롤링 할까요? (최대 400): "))

# 날짜 형식 변환 (URL에 사용될 형식)
def convert_date_format(date_str):
    return date_str.replace('.', '')

# 날짜 문자열을 datetime 객체로 변환
def str_to_datetime(date_str):
    return datetime.datetime.strptime(date_str, "%Y.%m.%d")

# datetime 객체를 문자열로 변환
def datetime_to_str(date_obj):
    return date_obj.strftime("%Y.%m.%d")

# 시작일과 종료일 사이의 기간을 n일 단위로 분할
def split_date_range(start_date_str, end_date_str, days=30):
    start_date = str_to_datetime(start_date_str)
    end_date = str_to_datetime(end_date_str)
    
    date_ranges = []
    current_start = start_date
    
    while current_start <= end_date:
        current_end = min(current_start + datetime.timedelta(days=days-1), end_date)
        date_ranges.append((datetime_to_str(current_start), datetime_to_str(current_end)))
        current_start = current_end + datetime.timedelta(days=1)
    
    return date_ranges

# 네이버 뉴스 본문 추출 함수
def extract_naver_news_content(url):
    try:
        headers = {'User-agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None, None, None, None
        
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        title = None
        content = None
        date = None
        press = None
        
        # 만약 연예 뉴스라면
        if "entertain" in response.url:
            title_element = soup.select_one(".end_tit")
            content_element = soup.select_one("#articeBody")
            date_element = soup.select_one("div.article_info > span > em")
            press_element = soup.select_one(".press_logo > img")
            
            if title_element:
                title = title_element.text.strip()
            if content_element:
                content = content_element.text.strip()
            if date_element:
                date = date_element.text.strip()
            if press_element:
                press = press_element.get('alt', '언론사 정보 없음')
        
        # 만약 스포츠 뉴스라면
        elif "sports" in response.url:
            title_element = soup.select_one("h4.title")
            content_element = soup.select_one("#newsEndContents")
            date_element = soup.select_one(".news_date")
            press_element = soup.select_one(".logo > img")
            
            if title_element:
                title = title_element.text.strip()
            if content_element:
                # 본문 내용안에 불필요한 div, p 제거
                for div in content_element.select("div"):
                    div.decompose()
                for p in content_element.select("p"):
                    p.decompose()
                content = content_element.text.strip()
            if date_element:
                date = date_element.text.strip()
            if press_element:
                press = press_element.get('alt', '언론사 정보 없음')
        
        # 일반 뉴스라면
        else:
            title_element = soup.select_one(".media_end_head_headline")
            content_element = soup.select_one("#newsct_article")
            date_element = soup.select_one("span.media_end_head_info_datestamp > em")
            press_element = soup.select_one(".media_end_head_top_logo > img")
            
            if title_element:
                title = title_element.text.strip()
            if content_element:
                content = content_element.text.strip()
            if date_element:
                date = date_element.text.strip()
            if press_element:
                press = press_element.get('alt', '언론사 정보 없음')
        
        # 기본값 설정
        if title is None:
            title = "제목 추출 실패"
        if content is None:
            content = "본문 추출 실패"
        if date is None:
            date = "날짜 추출 실패"
        if press is None:
            press = "언론사 정보 없음"
        
        return title, press, date, content
    
    except Exception as e:
        return None, None, None, None

# 데이터 저장용 리스트 초기화
data_list = []

# 날짜 범위 분할 (한 번에 30일씩)
date_ranges = split_date_range(start_date_str, end_date_str, 30)

for period_start, period_end in date_ranges:
    period_start_formatted = convert_date_format(period_start)
    period_end_formatted = convert_date_format(period_end)
    
    # 페이지 번호 순회
    for i in range(1, max_pages_per_period * 10, 10):
        
        # 네이버 뉴스 검색 URL (날짜 범위 지정)
        url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={keyword}&sort=0&photo=0&field=0&pd=3&ds={period_start}&de={period_end}&cluster_rank=32&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from{period_start_formatted}to{period_end_formatted},a:all&start={i}"
        
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        # 뉴스 기사 목록 추출
        news_items = soup.select("div.news_wrap.api_ani_send")
        
        # 뉴스 기사가 없으면 해당 페이지에서 크롤링 중단
        if not news_items:
            break
        
        for item in news_items:
            # 제목 추출
            title_element = item.select_one("a.news_tit")
            if not title_element:
                continue
            
            search_title = title_element.text.strip()
            news_url = title_element.get('href')
            
            # 원본 뉴스 URL 확인
            if "news.naver.com" in news_url:
                # 네이버 뉴스 본문 추출
                title, press, date, content = extract_naver_news_content(news_url)
                
                if title and content:  # 성공적으로 추출된 경우 데이터 추가
                    data_list.append([title, press, date, content, news_url])
            
            else:
                # 네이버 뉴스가 아닌 경우 검색 결과의 정보만 저장
                press_element = item.select_one("a.info.press")
                press = press_element.text.strip() if press_element else "언론사 정보 없음"
                
                date_element = item.select_one("span.info")
                date = date_element.text.strip() if date_element else "날짜 정보 없음"
                
                data_list.append([search_title, press, date, "네이버 뉴스 링크가 아님", news_url])
            
            # 과도한 요청 방지를 위한 딜레이
            time.sleep(0.5)

# 데이터프레임 생성 및 CSV 저장
df = pd.DataFrame(data_list, columns=["제목", "언론사", "날짜", "본문", "링크"])
file_name = f"{keyword}_{convert_date_format(start_date_str)}_{convert_date_format(end_date_str)}.csv"
df.to_csv(file_name, index=False, encoding='utf-8-sig')

print(f"크롤링이 완료되었습니다. 결과가 {file_name} 파일에 저장되었습니다.")
