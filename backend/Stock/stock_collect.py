import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import os

def get_stock_price_info(api_key, start_date=None, end_date=None, item_name=None, market_code=None, page_no=1, num_of_rows=10):
    """
    주식시세 정보를 가져오는 함수
    
    Parameters:
    - api_key: 공공데이터포털 API 인증키 (디코딩된 상태)
    - start_date: 조회 시작일(YYYYMMDD)
    - end_date: 조회 종료일(YYYYMMDD)
    - item_name: 종목명
    - market_code: 시장 구분 코드 (KOSPI, KOSDAQ, KONEX)
    - page_no: 페이지 번호
    - num_of_rows: 한 페이지 결과 수
    
    Returns:
    - DataFrame: 주식 시세 정보
    """
    url = "http://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"
    
    params = {
        'serviceKey': api_key,
        'numOfRows': num_of_rows,
        'pageNo': page_no,
        'resultType': 'xml'
    }
    
    if start_date:
        params['beginBasDt'] = start_date
    if end_date:
        params['endBasDt'] = end_date
    if item_name:
        params['itmsNm'] = item_name
    if market_code:
        params['mrktCtg'] = market_code
    
    try:
        response = requests.get(url, params=params)
        print(f"요청 URL: {response.url}")
        print(f"응답 상태 코드: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error: API 요청 실패 (상태 코드: {response.status_code})")
            print(f"응답 내용: {response.text[:500]}...")
            return None
        
        root = ET.fromstring(response.text)
        
        header = root.find('.//header')
        if header is not None:
            result_code = header.find('resultCode')
            result_msg = header.find('resultMsg')
            
            if result_code is not None and result_code.text != '00':
                print(f"API 오류: {result_code.text} - {result_msg.text if result_msg is not None else '알 수 없는 오류'}")
                return None
        
        items = root.findall('.//item')
        
        if not items:
            print("데이터가 없습니다.")
            return None
        
        data = []
        for item in items:
            row = {}
            for child in item:
                row[child.tag] = child.text.strip() if child.text else None
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

import os
from datetime import datetime

def save_to_csv(df, item_name, start_date, end_date, market_code, use_parent_dir=True):
    """
    DataFrame을 CSV 파일로 저장하는 함수
    
    Parameters:
    - df: 저장할 DataFrame
    - item_name: 종목명
    - start_date: 시작일
    - end_date: 종료일
    - market_code: 시장 구분 코드
    - use_parent_dir: True면 '../../CSV'에 저장, False면 현재 위치의 'CSV' 폴더에 저장
    
    Returns:
    - str: 저장된 파일 경로
    """
    # 파일명 생성
    if item_name:
        item_name_simple = item_name.replace(" ", "_")
    else:
        item_name_simple = "all_stocks"
    
    market_str = market_code if market_code else "all_markets"
    filename = f"{item_name_simple}_{start_date}_{end_date}_{market_str}.csv"

    # 저장 경로 설정
    if use_parent_dir:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'CSV'))
    else:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CSV'))

    # 디렉토리가 없으면 생성
    os.makedirs(save_dir, exist_ok=True)

    # 파일 경로 생성
    file_path = os.path.join(save_dir, filename)
    
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"파일이 저장되었습니다: {file_path}")
    return file_path
def get_data_by_date_range(api_key, start_date, end_date, item_name=None, market_code=None, num_of_rows=1000):
    """
    특정 기간에 대한 주식 데이터를 가져오는 함수 (페이지네이션 적용)
    
    Parameters:
    - api_key: API 인증키
    - start_date: 시작일 (YYYYMMDD)
    - end_date: 종료일 (YYYYMMDD)
    - item_name: 종목명
    - market_code: 시장 구분 코드 (KOSPI, KOSDAQ, KONEX)
    - num_of_rows: 한 페이지 결과 수
    
    Returns:
    - DataFrame: 해당 기간의 주식 데이터
    """
    print(f"{start_date} ~ {end_date} 데이터 수집 중...")
    
    all_data = []
    page_no = 1  # 초기 페이지 번호 설정
    
    while True:
        # 현재 페이지 데이터 요청
        df = get_stock_price_info(api_key, start_date, end_date, item_name, market_code, page_no=page_no, num_of_rows=num_of_rows)
        
        if df is None or df.empty:
            # 데이터가 없으면 루프 종료
            break
        
        all_data.append(df)  # 현재 페이지 데이터를 리스트에 추가
        print(f"페이지 {page_no} 데이터 수집 완료 (총 {len(df)} 행)")
        
        # 반환된 데이터 개수가 num_of_rows보다 적으면 마지막 페이지로 판단하고 종료
        if len(df) < num_of_rows:
            break
        
        # 다음 페이지로 이동
        page_no += 1
    
    if not all_data:
        return None
    
    # 모든 페이지 데이터를 하나의 DataFrame으로 결합
    return pd.concat(all_data, ignore_index=True)


def main():
    # 디코딩된 API 인증키 사용
    api_key = "4C1R2UncJxENVdWiSDvzwSIyPNyTQcX5/QO4O7XVNiliXmBif4rSp94tQTRowZOoZC3UjRdfeX3v+Cf+aoXuaA=="
    
    print("주식시세정보 수집 프로그램")
    print("-" * 50)
    
    # 시장 구분 선택
    print("\n시장 구분 선택:")
    print("1: KOSPI")
    print("2: KOSDAQ")
    print("3: KONEX")
    print("4: 전체")
    market_choice = input("선택: ")
    
    market_code = None
    if market_choice == "1":
        market_code = "KOSPI"
    elif market_choice == "2":
        market_code = "KOSDAQ"
    elif market_choice == "3":
        market_code = "KONEX"
    
    # 종목명 입력 (선택사항)
    item_name = input("\n종목명 (전체 종목을 원하면 빈칸): ")
    
    # 조회 기간 설정
    print("\n조회 기간 설정:")
    today = datetime.now()
    
    default_end_date = today.strftime("%Y%m%d")
    default_start_date = (today - timedelta(days=7)).strftime("%Y%m%d")
    
    start_date = input(f"시작일 (YYYYMMDD, 기본값: {default_start_date}): ") or default_start_date
    end_date = input(f"종료일 (YYYYMMDD, 기본값: {default_end_date}): ") or default_end_date
    
    # 데이터 수집
    df = get_data_by_date_range(api_key, start_date, end_date, item_name, market_code, num_of_rows=1000)
    
    # 수집된 데이터가 있는지 확인
    if df is not None and not df.empty:
        # 날짜 기준으로 정렬
        if 'basDt' in df.columns:
            df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')
            df = df.sort_values('basDt')
            df['basDt'] = df['basDt'].dt.strftime('%Y%m%d')
        
        print(f"\n총 {len(df)}개의 데이터를 수집했습니다.")
        
        # 컬럼명 확인 및 출력
        print("\n컬럼 목록:")
        for col in df.columns:
            print(f"- {col}")
        
        # 데이터 미리보기
        print("\n데이터 미리보기:")
        print(df.head())
        
        # CSV 파일로 저장
        save_to_csv(df, item_name, start_date, end_date, market_code)
    else:
        print("수집된 데이터가 없습니다.")

if __name__ == "__main__":
    main()