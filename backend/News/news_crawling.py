#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# 시작일과 종료일 사이의 기간을 n일 단위(기본 30일)로 분할
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

# 뉴스 본문 추출 함수 (다양한 뉴스 사이트 지원)
def extract_news_content(url):
    try:
        headers = {'User-agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None, None, None, None

        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        # 여러 선택자를 순차적으로 적용해 텍스트 추출 (img 태그는 alt 속성 읽기)
        def extract_text(soup, selectors):
            for sel in selectors:
                element = soup.select_one(sel)
                if element:
                    if element.name == "img":
                        text = element.get("alt", "").strip()
                    else:
                        text = element.get_text(strip=True)
                    if text:
                        return text
            return ""
        
        # 본문(content) 추출: 특정 선택자(#newsEndContents)인 경우 불필요한 태그 제거 후 처리
        def extract_content(soup, selectors):
            for sel in selectors:
                element = soup.select_one(sel)
                if element:
                    if sel == "#newsEndContents":
                        for tag in element.find_all(["div", "p", "script", "style"]):
                            tag.decompose()
                    text = element.get_text(strip=True)
                    if text:
                        return text
            return ""
        
        # 여러 선택자를 적용하여 데이터 추출 (네이버 뉴스 및 다른 뉴스 사이트 대응)
        title_selectors = [".media_end_head_headline", ".end_tit", "h4.title", 
                          "h1.article-title", "h1.headline", "h1.title", "h1", "title"]
        
        content_selectors = ["#newsct_article", "#articeBody", "#newsEndContents", 
                           ".article-content", ".article_body", ".article-text", ".article", 
                           "article", "#article", "#content", ".content", "#main-content"]
        
        date_selectors = ["span.media_end_head_info_datestamp > em", "div.article_info > span > em", 
                        ".news_date", ".article-date", ".date", "time", ".time", ".published"]
        
        press_selectors = [".media_end_head_top_logo > img", ".press_logo > img", ".logo > img", 
                         ".article-company", ".publisher", ".source", ".newspaper"]
        
        # 데이터 추출
        title = extract_text(soup, title_selectors)
        content = extract_content(soup, content_selectors)
        date = extract_text(soup, date_selectors)
        press = extract_text(soup, press_selectors)
        
        # 값이 없으면 기본 메시지 지정
        if not title:
            title = "제목 추출 실패"
        if not content:
            content = "본문 추출 실패"
        if not date:
            date = "날짜 추출 실패"
        if not press:
            press = "언론사 정보 없음"
        
        return title, press, date, content

    except Exception as e:
        print(f"뉴스 본문 추출 중 에러 발생 ({url}): {e}")
        return None, None, None, None

# 크롤링 결과를 저장할 리스트 초기화
data_list = []

# 날짜 범위를 지정한 일(기본 30일) 단위로 분할
date_ranges = split_date_range(start_date_str, end_date_str, days=30)

for period_start, period_end in date_ranges:
    period_start_formatted = convert_date_format(period_start)
    period_end_formatted = convert_date_format(period_end)
    
    # 페이지별로 크롤링(검색 결과 페이지의 start 파라미터는 1부터 10씩 증가)
    for i in range(1, max_pages_per_period * 10, 10):
        
        # 네이버 뉴스 검색 URL (날짜 범위 및 start 값 포함)
        url = (f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={keyword}"
               f"&sort=0&photo=0&field=0&pd=3&ds={period_start}&de={period_end}"
               f"&cluster_rank=32&mynews=0&office_type=0&office_section_code=0&news_office_checked="
               f"&nso=so:r,p:from{period_start_formatted}to{period_end_formatted},a:all&start={i}")
        
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        # 뉴스 기사 목록 추출
        news_items = soup.select("div.news_wrap.api_ani_send")
        
        # 뉴스 기사가 없으면 해당 페이지의 크롤링 중단
        if not news_items:
            break
        
        for item in news_items:
            title_element = item.select_one("a.news_tit")
            if not title_element:
                continue
            
            search_title = title_element.text.strip()
            news_url = title_element.get('href')
            
            # 검색 결과에서 기본 정보 가져오기
            press_element = item.select_one("a.info.press")
            search_press = press_element.text.strip() if press_element else "언론사 정보 없음"
            date_element = item.select_one("span.info")
            search_date = date_element.text.strip() if date_element else "날짜 정보 없음"
            
            try:
                # 모든 뉴스 URL에서 본문 추출 시도
                title, press, date, content = extract_news_content(news_url)
                
                # 본문 추출 성공 시 해당 정보 사용, 실패 시 검색 결과 정보 사용
                if title == "제목 추출 실패":
                    title = search_title
                if press == "언론사 정보 없음":
                    press = search_press
                if date == "날짜 추출 실패":
                    date = search_date
                
                # 제목이 없는 경우 또는 키워드가 포함되지 않은 경우 저장하지 않음
                if not title.strip() or keyword not in title:
                    print(f"키워드가 포함되지 않은 기사 건너뜀: {title[:30]}...")
                    continue
                
                # 데이터 저장
                data_list.append([title, press, date, content, news_url])
                print(f"기사 처리: {title[:30]}...")
                
            except Exception as e:
                # 예외 발생 시 검색 결과 정보로 저장 (제목이 없는 경우 또는 키워드가 포함되지 않은 경우 건너뜀)
                if not search_title.strip() or keyword not in search_title:
                    print(f"키워드가 포함되지 않은 기사 건너뜀: {search_title[:30]}...")
                    continue
                
                data_list.append([search_title, search_press, search_date, f"본문 추출 실패: {str(e)}", news_url])
                print(f"기사 처리 실패: {search_title[:30]}... - {str(e)}")
            
            # 과도한 요청 방지를 위한 딜레이
            time.sleep(0.5)

# 크롤링 데이터를 데이터프레임으로 변환 후 CSV 파일로 저장
df = pd.DataFrame(data_list, columns=["제목", "언론사", "날짜", "본문", "링크"])
file_name = f"{keyword}_{convert_date_format(start_date_str)}_{convert_date_format(end_date_str)}.csv"
df.to_csv(file_name, index=False, encoding='utf-8-sig')

print(f"크롤링이 완료되었습니다. 총 {len(data_list)}개의 기사를 수집했습니다.")
print(f"결과가 {file_name} 파일에 저장되었습니다.")
