"""
다중 종목 자동 매매 시스템 ver 1.0
- 단일 현금 자산 풀 공유
- 이동평균 크로스오버 전략 적용
- 3종목 동시 관리
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------
# 1. 시스템 초기화
# ---------------------------
class TradingSystem:
    def __init__(self, initial_capital=100000000):
        # 포트폴리오 초기화 (단일 현금 자산 + 종목별 보유량)
        self.portfolio = {
            'cash': initial_capital,  # 공유 현금 자산
            'holdings': {'삼성전자': 0, 'SK하이닉스': 0, 'NAVER': 0}  # 종목별 보유 수량
        }
        
        # 거래 기록 저장소
        self.trade_log = []
        
        # 종목별 데이터 로드
        self.stock_data = {
            '삼성전자': self.load_data('삼성전자_hist.csv'),
            'SK하이닉스': self.load_data('SK하이닉스_hist.csv'),
            'NAVER': self.load_data('NAVER_hist.csv')
        }

    def load_data(self, filename):
        """CSV 파일에서 주가 데이터 로드"""
        df = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
        df = df.sort_index()
        # 기술적 지표 계산
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        return df

# ---------------------------
# 2. 거래 엔진
# ---------------------------
    def execute_order(self, stock, action, price, quantity):
        """주문 실행 함수"""
        if action == 'buy':
            cost = price * quantity
            if self.portfolio['cash'] >= cost:
                self.portfolio['cash'] -= cost
                self.portfolio['holdings'][stock] += quantity
                self.record_trade(stock, '매수', price, quantity, cost)
        elif action == 'sell':
            if self.portfolio['holdings'][stock] >= quantity:
                revenue = price * quantity
                self.portfolio['cash'] += revenue
                self.portfolio['holdings'][stock] -= quantity
                self.record_trade(stock, '매도', price, quantity, revenue)

    def record_trade(self, stock, action, price, qty, amount):
        """거래 기록 저장"""
        log = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            '종목': stock,
            '행동': action,
            '가격': price,
            '수량': qty,
            '금액': amount,
            '현금잔고': self.portfolio['cash'],
            '보유량': self.portfolio['holdings'][stock]
        }
        self.trade_log.append(log)

# ---------------------------
# 3. 트레이딩 전략
# ---------------------------
    def generate_signal(self, stock):
        """이동평균 크로스오버 전략"""
        df = self.stock_data[stock]
        today = df.index[-1]
        
        # 전략 조건
        buy_cond = (df.loc[today, 'MA5'] > df.loc[today, 'MA20']) 
        sell_cond = (df.loc[today, 'MA5'] < df.loc[today, 'MA20'])
        
        if buy_cond:
            return 'buy'
        elif sell_cond:
            return 'sell'
        return 'hold'

# ---------------------------
# 4. 시뮬레이션 실행
# ---------------------------
    def run(self, start_date, end_date):
        """백테스팅 실행"""
        date_range = pd.date_range(start=start_date, end=end_date)
        
        for date in date_range:
            for stock in self.stock_data:
                # 해당 날짜 데이터 존재 여부 확인
                if date in self.stock_data[stock].index:
                    current_price = self.stock_data[stock].loc[date, 'Close']
                    signal = self.generate_signal(stock)
                    
                    # 전략 실행
                    if signal == 'buy':
                        available_qty = self.portfolio['cash'] // current_price
                        if available_qty > 0:
                            self.execute_order(stock, 'buy', current_price, available_qty)
                    elif signal == 'sell':
                        if self.portfolio['holdings'][stock] > 0:
                            self.execute_order(stock, 'sell', current_price, self.portfolio['holdings'][stock])
        
        # 최종 청산
        self.liquidate_all()

    def liquidate_all(self):
        """모든 포지션 청산"""
        for stock in self.portfolio['holdings']:
            if self.portfolio['holdings'][stock] > 0:
                current_price = self.stock_data[stock].iloc[-1]['Close']
                self.execute_order(stock, 'sell', current_price, self.portfolio['holdings'][stock])

# ---------------------------
# 5. 결과 분석
# ---------------------------
    def portfolio_value(self):
        """포트폴리오 가치 계산"""
        total = self.portfolio['cash']
        for stock, qty in self.portfolio['holdings'].items():
            last_price = self.stock_data[stock].iloc[-1]['Close']
            total += qty * last_price
        return total

    def show_results(self):
        """결과 리포트 출력"""
        print(f"최종 포트폴리오 가치: {self.portfolio_value():,}원")
        print("\n거래 내역:")
        for log in self.trade_log[-5:]:  # 최근 5건만 출력
            print(f"[{log['date']}] {log['종목']} {log['행동']} {log['수량']}주 @ {log['가격']:,}원")

# ---------------------------
# 실행 예시
# ---------------------------
if __name__ == "__main__":
    system = TradingSystem(initial_capital=100_000_000)
    system.run('2024-01-01', '2024-12-31')
    system.show_results()
