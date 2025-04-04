import requests
import pandas as pd
import xml.etree.ElementTree as ET
import os

def get_stock_price_info(api_key, start_date=None, end_date=None, item_name=None, market_code=None, page_no=1, num_of_rows=10):
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
        if response.status_code != 200:
            return None
        root = ET.fromstring(response.text)
        items = root.findall('.//item')
        if not items:
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
        return None

def save_to_csv(df, item_name, start_date, end_date, market_code, save_dir):
    # 파일명에 공백 대신 밑줄 사용하고, 종목코드는 넣지 않습니다.
    item_name_simple = item_name.replace(" ", "_") if item_name else "all_stocks"
    market_str = market_code if market_code else "all_markets"
    filename = f"{item_name_simple}_{start_date}_{end_date}_{market_str}.csv"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"Data saved to: {file_path}")
    return file_path

def get_data_by_date_range(api_key, start_date, end_date, item_name=None, market_code=None, num_of_rows=1000):
    all_data = []
    page_no = 1
    while True:
        df = get_stock_price_info(api_key, start_date, end_date, item_name, market_code, page_no=page_no, num_of_rows=num_of_rows)
        if df is None or df.empty:
            break
        all_data.append(df)
        if len(df) < num_of_rows:
            break
        page_no += 1
    if not all_data:
        return None
    return pd.concat(all_data, ignore_index=True)

def main():
    api_key = "4C1R2UncJxENVdWiSDvzwSIyPNyTQcX5/QO4O7XVNiliXmBif4rSp94tQTRowZOoZC3UjRdfeX3v+Cf+aoXuaA=="
    market_code = "KOSPI"  # 고정값
    save_dir = "./CSV"

    # 종목 리스트 (종목코드는 파일명에 포함하지 않습니다.)
    stock_list = [
        {"name": "삼성전자", "code": "005930"},
        {"name": "SK하이닉스", "code": "000660"},
        {"name": "현대차", "code": "005380"},
        {"name": "기아", "code": "000270"},
        {"name": "LIG넥스원", "code": "079550"},
        {"name": "삼성SDI", "code": "006400"},
        {"name": "셀트리온", "code": "068270"},
        {"name": "삼성바이오로직스", "code": "207940"},
        {"name": "KB금융", "code": "105560"},
        {"name": "신한지주", "code": "055550"},
        {"name": "LG전자", "code": "066570"},
        {"name": "아모레퍼시픽", "code": "090430"},
        {"name": "대한항공", "code": "003490"},
        {"name": "크래프톤", "code": "259960"},
        {"name": "삼성물산", "code": "028260"},
        {"name": "대우건설", "code": "047040"},
        {"name": "HMM", "code": "011200"},
        {"name": "한화솔루션", "code": "009830"},
        {"name": "NAVER", "code": "035420"},
        {"name": "KT", "code": "030200"}
    ]

    # 조회할 날짜 범위들
    date_ranges = [
        ("20200102", "20231231"),
        ("20240102", "20241231")
    ]

    # 각 종목별로 두 날짜 범위의 데이터를 수집하여 CSV 저장
    for stock in stock_list:
        for start_date, end_date in date_ranges:
            df = get_data_by_date_range(api_key, start_date, end_date, item_name=stock["name"], market_code=market_code)
            if df is not None and not df.empty:
                if 'basDt' in df.columns:
                    df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')
                    df = df.sort_values('basDt')
                    df['basDt'] = df['basDt'].dt.strftime('%Y%m%d')
                # 파일명에 종목 이름만 포함 (종목코드 제거)
                item_identifier = stock["name"]
                save_to_csv(df, item_name=item_identifier, start_date=start_date, end_date=end_date, market_code=market_code, save_dir=save_dir)

if __name__ == "__main__":
    main()
