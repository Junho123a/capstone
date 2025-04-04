#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import font_manager, rc
from sklearn.preprocessing import MinMaxScaler
from lime.lime_tabular import LimeTabularExplainer
import logging
from pathlib import Path
import argparse
import configparser
from datetime import datetime

class ProductionTester:
    def __init__(self, config_path='config.ini'):
        self._load_config(config_path)
        self._configure_logging()
        self._configure_font()
        self._init_paths()
        # 초기에는 모델을 로드하지 않음
        self.model = None
        self.scaler = None
        self.seq_length = None

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

    def _load_resources(self, csv_path=None):
        if csv_path:
            # 파일명에서 종목명 추출 (첫 번째 '_' 앞부분)
            stock_name = Path(csv_path).name.split('_')[0]
            model_path = f'{stock_name}/timesnet_model.h5'
            scaler_path = f'{stock_name}/timesnet_scaler.joblib'
        else:
            model_path = 'cnn_model.h5'
            scaler_path = 'scaler.joblib'
        
        required_files = {
            'model': (model_path, load_model),
            'scaler': (scaler_path, joblib.load)
        }
        
        resources = {}
        for name, (file, loader) in required_files.items():
            path = self.model_dir / file
            if not path.exists():
                raise FileNotFoundError(f"{name} 파일 없음: {path}")
            resources[name] = loader(path)
        
        if not isinstance(resources['scaler'], MinMaxScaler):
            raise TypeError("잘못된 스케일러 타입")
            
        return resources['model'], resources['scaler']


    def select_csv(self):
        files = list(self.csv_dir.glob('*.csv'))
        if not files:
            raise FileNotFoundError("CSV 파일 없음")
            
        self.logger.info("\n[테스트 가능 파일 목록]")
        for idx, f in enumerate(files, 1):
            self.logger.info(f"  {idx}. {f.name}")
            
        while True:
            choice = input(f"선택 (1-{len(files)}, 종료:0): ")
            if choice == '0':
                raise SystemExit("종료")
            try:
                choice = int(choice) - 1
                return files[choice]
            except (ValueError, IndexError):
                self.logger.error("유효한 숫자 입력 필요")

    def load_test_data(self, csv_path):
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='euc-kr')

        if 'clpr' not in df.columns or 'basDt' not in df.columns:
            raise ValueError("종가(clpr)와 기준일자(basDt) 컬럼 필수")
        
        df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')
        df = df[['basDt', 'clpr']].copy().dropna().sort_values('basDt')
        return df

    def generate_test_report(self):
        csv_path = self.select_csv()
        df = self.load_test_data(csv_path)
        
        # CSV 파일 선택 후 해당 종목에 맞는 모델 로드
        self.model, self.scaler = self._load_resources(csv_path)
        self.seq_length = self.model.input_shape[1]
        
        scaled = self.scaler.transform(df[['clpr']].values)
        X_test = np.array([scaled[i:i+self.seq_length] 
                         for i in range(len(scaled)-self.seq_length)])
        y_test = scaled[self.seq_length:]
        
        preds = self.model.predict(X_test, verbose=1)
        actual = self.scaler.inverse_transform(y_test)
        preds = self.scaler.inverse_transform(preds)
        
        report_df, metrics = self._analyze_results(actual, preds)
        self._save_artifacts(report_df, metrics)
        self._generate_visualization(df, actual, preds, report_df, metrics)
        
        X_test_flat = X_test.reshape(X_test.shape[0], self.seq_length)
        self._explain_with_lime(X_test_flat, X_test_flat[0])
        
        self.logger.info("\n[테스트 완료]")
        self.logger.info(f"- 리포트: {self.result_dir / 'stock_report.csv'}")
        self.logger.info(f"- 시각화: {self.result_dir / 'stock_analysis.png'}")
        self.logger.info(f"- LIME 설명: {self.result_dir / 'lime_explanation.txt'}")

    def _analyze_results(self, actual, preds):
        report_df = pd.DataFrame({
            'Actual': actual.flatten(),
            'Predicted': preds.flatten(),
            'Error%': ((actual - preds)/actual * 100).flatten()
        }).copy()

        metrics = {
            'MSE': mean_squared_error(actual, preds),
            'MAE': mean_absolute_error(actual, preds),
            'R2': r2_score(actual, preds),
            'Accuracy': 100 - np.mean(np.abs(report_df['Error%']))
        }

        actual_chg = np.sign(np.diff(actual.flatten()))
        pred_chg = np.sign(np.diff(preds.flatten()))
        metrics['Direction'] = np.mean(actual_chg == pred_chg) * 100

        periods = {
            '1개월': 21, 
            '3개월': 63,
            '6개월': 126
        }
        
        metrics['Achievement'] = {}
        for period, days in periods.items():
            if len(actual) > days:
                valid_actual = actual[days:]
                valid_preds = preds[:-days]
                achieved = np.abs(valid_actual - valid_preds) <= valid_actual * float(self.config.get('target_error_rate', 0.05))
                metrics['Achievement'][period] = np.mean(achieved) * 100
            else:
                metrics['Achievement'][period] = np.nan

        return report_df, metrics

    def _save_artifacts(self, report_df, metrics):
        report_df.to_csv(self.result_dir / 'stock_report.csv', index=False, encoding='utf-8-sig')
        
        with open(self.result_dir / 'summary.txt', 'w', encoding='utf-8') as f:
            f.write(f"""cnn_model 모델 성능 리포트
========================
- 데이터 포인트: {len(report_df):,}개
- 평균 제곱 오차(MSE): {metrics['MSE']:.2f}
- 평균 절대 오차(MAE): {metrics['MAE']:.2f}
- 결정계수(R2): {metrics['R2']:.2f}
- 방향 예측 정확도: {metrics['Direction']:.2f}%
- 평균 정확도: {metrics['Accuracy']:.2f}%

[5% 목표 달성률]
"""
            + '\n'.join([f"- {k}: {v:.2f}%" for k,v in metrics['Achievement'].items()]))

    def _generate_visualization(self, df, actual, preds, report_df, metrics):
        plt.figure(figsize=(24, 20))
        plt.suptitle("주가 예측 분석 리포트", y=1.02, fontsize=16)

        plt.subplot(4, 2, (1, 2))
        plt.plot(df['basDt'], df['clpr'], label='실제 주가', color='blue')
        plt.plot(df['basDt'][self.seq_length:], preds, label='예측 주가', color='red', linestyle='--')
        plt.title('전체 주가 추이 및 예측')
        plt.xlabel('날짜')
        plt.ylabel('주가')
        plt.legend()

        plt.subplot(4, 2, 3)
        sns.histplot(report_df['Error%'], kde=True, bins=30)
        plt.xlabel('오차율 (%)')
        plt.title('예측 오차 분포')

        plt.subplot(4, 2, 4)
        window = min(60, len(actual)//4)
        plt.plot(df['basDt'][self.seq_length:], pd.Series(actual.flatten()).rolling(window).mean(), label='실제')
        plt.plot(df['basDt'][self.seq_length:], pd.Series(preds.flatten()).rolling(window).mean(), label='예측')
        plt.title(f'{window}일 이동평균 비교')
        plt.legend()

        plt.subplot(4, 2, 5)
        achieved = [v for v in metrics['Achievement'].values() if not np.isnan(v)]
        labels = [k for k,v in metrics['Achievement'].items() if not np.isnan(v)]
        plt.bar(labels, achieved, color=['#4CAF50','#2196F3','#FFC107'])
        plt.axhline(100, color='red', linestyle='--', alpha=0.5)
        plt.ylabel('달성률 (%)')
        plt.title('기간별 목표 달성 현황')

        plt.subplot(4, 2, (6, 8))
        df['pct_change'] = df['clpr'].pct_change()
        significant_changes = df[abs(df['pct_change']) > df['pct_change'].std() * 2]
        plt.bar(significant_changes['basDt'], significant_changes['pct_change'] * 100, 
                color=np.where(significant_changes['pct_change'] > 0, 'g', 'r'))
        plt.title('주요 주가 변동 날짜')
        plt.xlabel('날짜')
        plt.ylabel('변동률 (%)')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(self.result_dir / 'stock_analysis.png', dpi=120, bbox_inches='tight')
        plt.close()

    def _explain_with_lime(self, training_data, x_instance):
        feature_names = [f"t-{self.seq_length - i}" for i in range(self.seq_length)]
        
        explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            mode="regression",
            verbose=True
        )
        
        def predict_fn(data):
            data_reshaped = data.reshape(-1, self.seq_length, 1)
            preds = self.model.predict(data_reshaped, verbose=0)
            return preds.flatten()

        explanation = explainer.explain_instance(
            x_instance,
            predict_fn,
            num_features=self.seq_length,
            num_samples=1000
        )
        
        lime_path = self.result_dir / 'lime_explanation.txt'
        with open(lime_path, 'w', encoding='utf-8') as f:
            f.write("선택된 인스턴스에 대한 LIME 설명\n")
            f.write("특성\t가중치\n")
            for feature, weight in explanation.as_list():
                f.write(f"{feature}\t{weight:.4f}\n")
                
    def run(self):
        try:
            self.generate_test_report()
        except Exception as e:
            self.logger.exception(f"\n오류 발생: {str(e)}")
        finally:
            input("\n엔터 키를 눌러 종료...")

def parse_arguments():
    parser = argparse.ArgumentParser(description='주가 예측 모델 테스터')
    parser.add_argument('--config', type=str, default='config.ini', help='설정 파일 경로')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    tester = ProductionTester(config_path=args.config)
    tester.run()
