import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(ticker, start_date, end_date):
    """Yahoo Finance에서 주가 데이터 가져오기"""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# 애플 주가 데이터 가져오기 (미국 주식)
ticker = "AAPL"  # 애플 티커
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1년치 데이터

stock_data = get_stock_data(ticker, start_date, end_date)
stock_data = stock_data.reset_index()
stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')

# 주가 데이터 저장
stock_data.to_csv("apple_stock.csv", index=False)
print("주가 데이터가 저장되었습니다.")
