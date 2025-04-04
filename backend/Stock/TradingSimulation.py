#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import font_manager, rc
from lime.lime_tabular import LimeTabularExplainer
import logging
from pathlib import Path
import argparse
import configparser
from datetime import datetime
import traceback
from tqdm import tqdm

class RoboAdvisor:
    def __init__(self, config_path='config.ini'):
        self._load_config(config_path)
        self._configure_logging()
        self._configure_font()
        self._init_paths()
        
        # 포트폴리오 관련 변수
        self.initial_amount = None
        self.current_amount = None
        self.cash = None
        self.portfolio = {}
        self.transactions = []
        self.start_date = None
        self.end_date = None
        
        # 종목별 데이터 및 모델
        self.stock_data = {}
        self.stock_models = {}
        self.stock_scalers = {}

    def _load_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        self.config = config['DEFAULT']

    def _configure_logging(self):
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)
        self.logger = logging.getLogger(__name__)

    def _configure_font(self):
        try:
            font_path = self.config.get('font_path', "C:/Windows/Fonts/malgun.ttf")
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            rc('font', family=font_name)
        except Exception as e:
            self.logger.warning(f"Font configuration failed: {e}. Using default font.")
            rc('font', family='DejaVu Sans')

    def _init_paths(self):
        self.base_dir = Path(self.config.get('base_dir', Path(__file__).parent.parent.parent))
        self.model_dir = self.base_dir / 'saved_models'
        self.csv_dir = self.base_dir / 'CSV'
        self.result_dir = self.base_dir / 'results'
        self.result_dir.mkdir(exist_ok=True)

    # 모델 로드 부분에서 인코딩 지정 예시
    def _load_stock_model(self, stock_name):
        try:
            # 파일 경로
            model_path = self.model_dir / f'{stock_name}/timesnet_model.h5'
            scaler_path = self.model_dir / f'{stock_name}/timesnet_scaler.joblib'
            
            # 파일 존재 확인
            if not model_path.exists() or not scaler_path.exists():
                self.logger.warning(f"{stock_name} 모델 또는 스케일러 파일 없음")
                return False
            
            # 인코딩 문제가 있는 경우 다음과 같이 수정
            # 1. 모델 로드 방식에 따라 인코딩 파라미터 추가
            model = load_model(model_path)  # Keras 모델은 인코딩 문제가 없음
            scaler = joblib.load(scaler_path)  # joblib은 바이너리로 저장하므로 인코딩 문제가 없음
            
            self.stock_models[stock_name] = model
            self.stock_scalers[stock_name] = scaler
            
            return True
        except UnicodeDecodeError as e:
            # 특정 인코딩 지정 시도
            self.logger.error(f"{stock_name} 모델 로드 실패 (인코딩 문제): {e}")
            self.logger.info(f"{stock_name} 모델 CP949 인코딩으로 다시 시도")
            # 여기에 대체 로딩 로직 추가
            return False
        except Exception as e:
            self.logger.error(f"{stock_name} 모델 로드 실패: {e}")
            return False

    def load_stock_data(self):
        """주식 데이터 파일 로드"""
        csv_files = list(self.csv_dir.glob('*_20240102_20241231_KOSPI.csv'))
        
        if not csv_files:
            raise FileNotFoundError("CSV 파일을 찾을 수 없습니다.")
        
        self.logger.info(f"총 {len(csv_files)}개의 종목 데이터를 로드합니다.")
        
        for csv_path in tqdm(csv_files, desc="데이터 로드 중"):
            stock_name = csv_path.name.split('_')[0]
            try:
                # CSV 파일 로드 및 전처리
                df = self._load_csv_file(csv_path)
                df = self._preprocess_stock_data(df)
                self.stock_data[stock_name] = df
                
                # 관련 모델 및 스케일러 로드
                self._load_stock_model(stock_name)
                
            except Exception as e:
                self.logger.error(f"{stock_name} 데이터 로드 실패: {e}")

    def _load_csv_file(self, csv_path):
        """CSV 파일 로드 함수"""
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='euc-kr')
        
        if 'basDt' not in df.columns or 'clpr' not in df.columns:
            raise ValueError(f"필수 열(basDt, clpr)이 없습니다: {csv_path}")
        
        return df

    def _preprocess_stock_data(self, df):
        """주식 데이터 전처리 함수"""
        # 날짜 형식 변환
        df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')
        
        # 필요 컬럼만 선택
        processed_df = df[['basDt', 'clpr', 'mkp', 'hipr', 'lopr', 'trqu']].copy()
        
        # NaN 제거
        processed_df = processed_df.dropna()
        
        # 날짜 기준 정렬
        processed_df = processed_df.sort_values('basDt')
        
        # 인덱스 리셋
        processed_df = processed_df.reset_index(drop=True)
        
        return processed_df

    def setup_portfolio(self):
        """포트폴리오 초기 설정"""
        # 초기 투자 금액 입력
        while True:
            try:
                amount_input = input("초기 투자 금액을 입력하세요 (단위: 원): ")
                self.initial_amount = int(amount_input.replace(',', ''))
                if self.initial_amount <= 0:
                    print("0보다 큰 금액을 입력하세요.")
                    continue
                break
            except ValueError:
                print("올바른 숫자 형식으로 입력하세요.")
        
        # 초기 현금 설정
        self.current_amount = self.initial_amount
        self.cash = self.initial_amount
        
        # 분석 기간 설정
        while True:
            try:
                start_date_str = input("분석 시작 날짜를 입력하세요 (YYYY-MM-DD): ")
                self.start_date = pd.to_datetime(start_date_str)
                
                end_date_str = input("분석 종료 날짜를 입력하세요 (YYYY-MM-DD): ")
                self.end_date = pd.to_datetime(end_date_str)
                
                if self.start_date >= self.end_date:
                    print("종료 날짜는 시작 날짜보다 나중이어야 합니다.")
                    continue
                
                # 유효한 날짜 범위 확인
                valid_dates = False
                for stock_name, df in self.stock_data.items():
                    if self.start_date in df['basDt'].values and self.end_date in df['basDt'].values:
                        valid_dates = True
                        break
                
                if not valid_dates:
                    print("입력한 날짜 범위에 데이터가 없습니다. 다른 날짜를 입력하세요.")
                    continue
                
                break
            except ValueError as e:
                print(f"올바른 날짜 형식으로 입력하세요: {e}")

    def predict_stock_trends(self, current_date):
        """각 종목별 주가 움직임 예측"""
        prediction_results = {}
        
        for stock_name, df in self.stock_data.items():
            if stock_name not in self.stock_models:
                continue
            
            try:
                # 현재 날짜까지의 데이터만 사용
                df_until_date = df[df['basDt'] <= current_date].copy()
                
                model = self.stock_models[stock_name]
                scaler = self.stock_scalers[stock_name]
                
                # 모델의 입력 형태 확인
                seq_length = model.input_shape[1]
                
                # 데이터 길이 확인
                available_data = df_until_date['clpr'].values
                
                # 데이터 패딩 적용
                if len(available_data) < seq_length:
                    self.logger.warning(f"{stock_name} 데이터가 부족합니다. 패딩 적용 ({len(available_data)}/{seq_length})")
                    
                    # 부족한 데이터 개수 계산
                    padding_needed = seq_length - len(available_data)
                    
                    # 옵션 1: 첫 번째 값으로 패딩 (또는 다른 적절한 값으로 대체 가능)
                    padding_value = available_data[0]
                    padded_data = np.full(padding_needed, padding_value)
                    
                    # 패딩된 데이터와 원본 데이터 연결
                    padded_sequence = np.concatenate([padded_data, available_data])
                else:
                    # 충분한 데이터가 있으면 마지막 seq_length개만 사용
                    padded_sequence = available_data[-seq_length:]
                
                # 패딩된 데이터로 예측 수행
                recent_data = padded_sequence.reshape(-1, 1)
                scaled_data = scaler.transform(pd.DataFrame(recent_data, columns=['clpr']))
                X_pred = scaled_data.reshape(1, seq_length, 1)
                scaled_prediction = model.predict(X_pred, verbose=0)
                
                # 원래 스케일로 변환
                prediction = scaler.inverse_transform(scaled_prediction)
                
                # 결과 저장
                current_price = df_until_date['clpr'].iloc[-1]
                predicted_price = prediction[0][0]
                pct_change = (predicted_price - current_price) / current_price * 100
                
                prediction_results[stock_name] = {
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'pct_change': pct_change,
                    'padded': len(available_data) < seq_length  # 패딩 적용 여부 표시
                }
                
            except Exception as e:
                self.logger.error(f"{stock_name} 예측 실패: {e}")
        
        return prediction_results


    def allocate_portfolio(self, prediction_results, current_date):
        """예측 결과를 바탕으로 포트폴리오 배분"""
        # 현재 포트폴리오 평가
        self._update_portfolio_value(current_date)
        
        # 상승 예측 종목 필터링 (임계값 이상)
        positive_threshold = float(self.config.get('positive_threshold', 0.5))
        rising_stocks = {k: v for k, v in prediction_results.items() 
                        if v['pct_change'] > positive_threshold}
        
        # 상승 예측 종목이 없으면 전량 매도하고 현금 보유
        if not rising_stocks:
            self.logger.info(f"상승 예측 종목이 없습니다. 전량 매도하고 현금으로 전환합니다.")
            self._sell_all_stocks(current_date)
            return
        
        # 상승률에 비례한 가중치 계산
        total_weight = sum(data['pct_change'] for data in rising_stocks.values())
        weights = {stock: data['pct_change'] / total_weight 
                for stock, data in rising_stocks.items()}
        
        # 목표 포트폴리오 계산
        target_portfolio = {}
        for stock, weight in weights.items():
            target_portfolio[stock] = self.current_amount * weight
        
        # 현재 보유 종목 중 목표 포트폴리오에 없는 종목 매도
        for stock in list(self.portfolio.keys()):
            if stock not in target_portfolio and self.portfolio[stock] > 0:
                self._sell_stock(stock, self.portfolio[stock], current_date)
        
        # 목표 포트폴리오에 따라 매수/매도 조정
        for stock, target_amount in target_portfolio.items():
            if stock not in self.portfolio:
                self.portfolio[stock] = 0
            
            current_price = prediction_results[stock]['current_price']
            current_value = self.portfolio[stock] * current_price
            
            # 목표 금액과 현재 금액의 차이
            diff_amount = target_amount - current_value
            
            # 차이에 따라 매수 또는 매도
            if diff_amount > 0 and self.cash >= diff_amount:
                # 매수할 수량 계산
                quantity_to_buy = int(diff_amount / current_price)
                if quantity_to_buy > 0:
                    self._buy_stock(stock, quantity_to_buy, current_date)
            elif diff_amount < 0:
                # 매도할 수량 계산
                quantity_to_sell = int(abs(diff_amount) / current_price)
                if quantity_to_sell > 0 and self.portfolio[stock] >= quantity_to_sell:
                    self._sell_stock(stock, quantity_to_sell, current_date)

    def _update_portfolio_value(self, current_date):
        """현재 포트폴리오 가치 업데이트"""
        portfolio_value = self.cash
        
        for stock, quantity in self.portfolio.items():
            if quantity <= 0:
                continue
                
            try:
                # 해당 날짜의 주가 확인
                df = self.stock_data[stock]
                current_row = df[df['basDt'] == current_date]
                
                if current_row.empty:
                    # 해당 날짜의 데이터가 없으면 가장 가까운 이전 날짜 사용
                    current_row = df[df['basDt'] < current_date].iloc[-1:]
                
                if not current_row.empty:
                    current_price = current_row['clpr'].values[0]
                    stock_value = quantity * current_price
                    portfolio_value += stock_value
            except Exception as e:
                self.logger.error(f"{stock} 가치 업데이트 실패: {e}")
        
        self.current_amount = portfolio_value

    def _buy_stock(self, stock_name, quantity, current_date):
        """주식 매수 처리"""
        # 해당 종목의 현재 주가 확인
        df = self.stock_data[stock_name]
        current_row = df[df['basDt'] == current_date]
        
        if current_row.empty:
            self.logger.warning(f"{current_date}에 {stock_name} 데이터 없음. 거래 취소")
            return
            
        price = current_row['clpr'].values[0]
        amount = price * quantity
        
        # 현금이 부족하면 가능한 만큼만 매수
        if amount > self.cash:
            quantity = int(self.cash / price)
            amount = price * quantity
            
        if quantity <= 0:
            return
            
        # 거래 기록
        transaction = {
            'date': current_date.strftime('%Y-%m-%d'),
            'stock': stock_name,
            'action': 'BUY',
            'quantity': quantity,
            'price': price,
            'amount': amount
        }
        self.transactions.append(transaction)
        
        # 포트폴리오 업데이트
        self.portfolio[stock_name] = self.portfolio.get(stock_name, 0) + quantity
        self.cash -= amount
        
        self.logger.info(f"매수: {stock_name} {quantity}주 @ {price:,.0f}원")

    def _sell_stock(self, stock_name, quantity, current_date):
        """주식 매도 처리"""
        # 해당 종목의 현재 주가 확인
        df = self.stock_data[stock_name]
        current_row = df[df['basDt'] == current_date]
        
        if current_row.empty:
            self.logger.warning(f"{current_date}에 {stock_name} 데이터 없음. 거래 취소")
            return
            
        price = current_row['clpr'].values[0]
        
        # 보유 수량보다 많이 매도할 수 없음
        if quantity > self.portfolio.get(stock_name, 0):
            quantity = self.portfolio.get(stock_name, 0)
            
        if quantity <= 0:
            return
            
        amount = price * quantity
        
        # 거래 기록
        transaction = {
            'date': current_date.strftime('%Y-%m-%d'),
            'stock': stock_name,
            'action': 'SELL',
            'quantity': quantity,
            'price': price,
            'amount': amount
        }
        self.transactions.append(transaction)
        
        # 포트폴리오 업데이트
        self.portfolio[stock_name] -= quantity
        self.cash += amount
        
        self.logger.info(f"매도: {stock_name} {quantity}주 @ {price:,.0f}원")

    def _sell_all_stocks(self, current_date):
        """모든 주식 매도"""
        for stock, quantity in list(self.portfolio.items()):
            if quantity > 0:
                self._sell_stock(stock, quantity, current_date)

    def _get_trading_days(self):
        """거래일 목록 가져오기"""
        # 모든 종목의 거래일 합집합
        all_dates = set()
        for stock, df in self.stock_data.items():
            dates = df[(df['basDt'] >= self.start_date) & 
                       (df['basDt'] <= self.end_date)]['basDt'].tolist()
            all_dates.update(dates)
        
        # 날짜순 정렬
        trading_days = sorted(list(all_dates))
        return trading_days

    def _filter_data_until_date(self, current_date):
        """현재 날짜까지의 데이터만 필터링"""
        filtered_data = {}
        for stock, df in self.stock_data.items():
            filtered_data[stock] = df[df['basDt'] <= current_date].copy()
        return filtered_data

    def _calculate_portfolio_value(self, current_date):
        """포트폴리오 가치 계산"""
        portfolio_value = self.cash
        holdings = {}
        
        for stock, quantity in self.portfolio.items():
            if quantity <= 0:
                continue
                
            try:
                # 해당 날짜의 주가 확인
                df = self.stock_data[stock]
                current_row = df[df['basDt'] == current_date]
                
                if current_row.empty:
                    # 해당 날짜의 데이터가 없으면 가장 가까운 이전 날짜 사용
                    current_row = df[df['basDt'] < current_date].iloc[-1:]
                
                if not current_row.empty:
                    current_price = current_row['clpr'].values[0]
                    stock_value = quantity * current_price
                    portfolio_value += stock_value
                    holdings[stock] = {
                        'quantity': quantity,
                        'price': current_price,
                        'value': stock_value
                    }
            except Exception as e:
                self.logger.error(f"{stock} 가치 계산 실패: {e}")
        
        return portfolio_value, holdings

    def _get_current_holdings(self):
        """현재 보유 종목 정보"""
        holdings = {}
        for stock, quantity in self.portfolio.items():
            if quantity > 0:
                holdings[stock] = quantity
        return holdings

    def run_backtest(self):
        """백테스트 실행"""
        # 거래일 목록 생성
        trading_days = self._get_trading_days()
        
        if not trading_days:
            self.logger.error("해당 기간에 거래일이 없습니다.")
            return
        
        self.logger.info(f"백테스트 시작: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')} ({len(trading_days)}일)")
        
        # 결과 저장용 변수
        results = []
        
        # 초기화
        self.portfolio = {}
        self.cash = self.initial_amount
        self.transactions = []
        
        # 각 거래일마다 시뮬레이션
        for current_date in tqdm(trading_days, desc="백테스트 진행 중"):
            # 예측 및 포트폴리오 조정
            predictions = self.predict_stock_trends(current_date)
            self.allocate_portfolio(predictions, current_date)
            
            # 포트폴리오 가치 계산 및 결과 저장
            portfolio_value, holdings = self._calculate_portfolio_value(current_date)
            
            results.append({
                'date': current_date,
                'cash': self.cash,
                'portfolio_value': portfolio_value,
                'holdings': holdings
            })
        
        # 결과 저장 및 시각화
        self._save_backtest_results(results)
        self._plot_backtest_results(results)
        self._save_transactions()
        
        # 결과 요약 출력
        self._print_summary(results)

    def _save_backtest_results(self, results):
        """백테스트 결과를 CSV로 저장"""
        # 기본 결과 데이터
        result_df = pd.DataFrame([
            {
                'date': r['date'],
                'portfolio_value': r['portfolio_value'],
                'cash': r['cash'],
                'stock_value': r['portfolio_value'] - r['cash']
            } for r in results
        ])
        
        result_df.to_csv(self.result_dir / 'backtest_results.csv', index=False, encoding='utf-8-sig')
        
        # 일별 종목 보유 현황
        holdings_data = []
        for r in results:
            base_row = {'date': r['date']}
            for stock, details in r['holdings'].items():
                base_row[f'{stock}_quantity'] = details['quantity']
                base_row[f'{stock}_value'] = details['value']
            holdings_data.append(base_row)
        
        holdings_df = pd.DataFrame(holdings_data)
        holdings_df.to_csv(self.result_dir / 'holdings_history.csv', index=False, encoding='utf-8-sig')

    def _save_transactions(self):
        """거래 내역을 CSV로 저장"""
        if not self.transactions:
            self.logger.warning("거래 내역이 없습니다.")
            return
            
        tx_df = pd.DataFrame(self.transactions)
        tx_df.to_csv(self.result_dir / 'transactions.csv', index=False, encoding='utf-8-sig')
        
        # 거래 요약 (종목별)
        summary = {}
        for tx in self.transactions:
            stock = tx['stock']
            action = tx['action']
            quantity = tx['quantity']
            amount = tx['amount']
            
            if stock not in summary:
                summary[stock] = {'BUY': 0, 'SELL': 0, 'BUY_AMOUNT': 0, 'SELL_AMOUNT': 0}
                
            summary[stock][action] += quantity
            summary[stock][f'{action}_AMOUNT'] += amount
        
        summary_data = []
        for stock, data in summary.items():
            net_quantity = data['BUY'] - data['SELL']
            net_amount = data['SELL_AMOUNT'] - data['BUY_AMOUNT']
            profit_pct = (net_amount / data['BUY_AMOUNT'] * 100) if data['BUY_AMOUNT'] > 0 else 0
            
            summary_data.append({
                'stock': stock,
                'total_buy': data['BUY'],
                'total_sell': data['SELL'],
                'net_quantity': net_quantity,
                'buy_amount': data['BUY_AMOUNT'],
                'sell_amount': data['SELL_AMOUNT'],
                'net_amount': net_amount,
                'profit_pct': profit_pct
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.result_dir / 'transaction_summary.csv', index=False, encoding='utf-8-sig')

    def _plot_backtest_results(self, results):
        """백테스트 결과 시각화"""
        # 기본 데이터 준비
        dates = [r['date'] for r in results]
        portfolio_values = [r['portfolio_value'] for r in results]
        cash_values = [r['cash'] for r in results]
        
        # 수익률 계산
        initial_value = self.initial_amount
        returns = [(v/initial_value-1)*100 for v in portfolio_values]
        
        plt.figure(figsize=(15, 20))
        
        # 1. 포트폴리오 가치 추이
        plt.subplot(3, 1, 1)
        plt.plot(dates, portfolio_values, label='포트폴리오 가치', linewidth=2)
        plt.plot(dates, cash_values, label='현금', linewidth=2, linestyle='--')
        plt.title('포트폴리오 가치 추이', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        
        # 2. 수익률 추이
        plt.subplot(3, 1, 2)
        plt.plot(dates, returns, label='수익률(%)', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('투자 수익률 (%)', fontsize=14)
        plt.grid(alpha=0.3)
        
        # 3. 자산 배분 (마지막 날 기준)
        plt.subplot(3, 1, 3)
        last_result = results[-1]
        holdings = last_result['holdings']
        
        if holdings:
            labels = ['현금'] + list(holdings.keys())
            sizes = [last_result['cash']] + [h['value'] for h in holdings.values()]
            
            if sum(sizes) > 0:  # 자산이 있는 경우에만 그래프 생성
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                plt.axis('equal')
                plt.title('최종 자산 배분', fontsize=14)
            else:
                plt.text(0.5, 0.5, '자산 없음', horizontalalignment='center', fontsize=14)
        else:
            plt.pie([1], labels=['현금'], autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('최종 자산 배분 (현금 100%)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.result_dir / 'backtest_performance.png', dpi=100)
        
        # 추가 시각화: 종목별 보유 변화
        self._plot_holdings_history(results)

    def _plot_holdings_history(self, results):
        """종목별 보유 변화 시각화"""
        # 모든 종목 추출
        all_stocks = set()
        for r in results:
            all_stocks.update(r['holdings'].keys())
        
        if not all_stocks:
            return
            
        # 날짜 데이터 추출 - 이 부분이 추가됨
        dates = [r['date'] for r in results]
        
        # 종목이 많으면 여러 그래프로 분할
        max_stocks_per_plot = 5
        stock_groups = [list(all_stocks)[i:i+max_stocks_per_plot] 
                        for i in range(0, len(all_stocks), max_stocks_per_plot)]
        
        for group_idx, stock_group in enumerate(stock_groups):
            plt.figure(figsize=(15, 10))
            
            # 종목별 보유 가치 변화
            for stock in stock_group:
                stock_values = []
                for r in results:
                    if stock in r['holdings']:
                        stock_values.append(r['holdings'][stock]['value'])
                    else:
                        stock_values.append(0)
                
                plt.plot(dates, stock_values, label=stock, linewidth=2)
            
            plt.title('종목별 보유 가치 변화', fontsize=14)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.savefig(self.result_dir / f'holdings_history_{group_idx+1}.png', dpi=100)
            plt.close()



    def _print_summary(self, results):
        """결과 요약 출력"""
        if not results:
            return
            
        initial_value = self.initial_amount
        final_value = results[-1]['portfolio_value']
        total_return = (final_value / initial_value - 1) * 100
        
        # 연간 수익률 계산
        start_date = results[0]['date']
        end_date = results[-1]['date']
        years = (end_date - start_date).days / 365.25
        annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
        
        self.logger.info("\n======= 백테스트 결과 요약 =======")
        self.logger.info(f"초기 투자금: {initial_value:,.0f}원")
        self.logger.info(f"최종 자산: {final_value:,.0f}원")
        self.logger.info(f"총 수익: {final_value-initial_value:,.0f}원 ({total_return:.2f}%)")
        self.logger.info(f"연간 수익률: {annual_return:.2f}%")
        self.logger.info(f"거래 횟수: {len(self.transactions)}회")
        self.logger.info(f"최종 현금 비중: {results[-1]['cash']/final_value*100:.2f}%")
        self.logger.info("===================================")

    def run(self):
        """로보 어드바이저 실행"""
        try:
            self.logger.info("로보 어드바이저 시작")
            
            # 주식 데이터 로드
            self.load_stock_data()
            
            if not self.stock_data:
                self.logger.error("로드된 주식 데이터가 없습니다.")
                return
                
            # 포트폴리오 설정
            self.setup_portfolio()
            
            # 백테스트 실행
            self.run_backtest()
            
            self.logger.info("로보 어드바이저 종료")
            
        except Exception as e:
            self.logger.exception(f"오류 발생: {str(e)}")
            traceback.print_exc()
        finally:
            input("\n엔터 키를 눌러 종료...")

def parse_arguments():
    parser = argparse.ArgumentParser(description='로보 어드바이저')
    parser.add_argument('--config', type=str, default='config.ini', help='설정 파일 경로')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    advisor = RoboAdvisor(config_path=args.config)
    advisor.run()
