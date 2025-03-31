import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

class ValidationEngine:
    """강화된 검증 엔진"""
    def __init__(self):
        self.model, self.scaler, self.base_dir = self._load_resources()
        self.csv_dir = os.path.abspath(os.path.join(self.base_dir, '..', '..', 'CSV'))
        self.results_dir = os.path.abspath(os.path.join(self.base_dir, '..', '..', 'results'))
        os.makedirs(self.results_dir, exist_ok=True)

    def _load_resources(self):
        """리소스 로드 로직 강화"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, '..', '..', 'saved_models')
        
        # 모델 및 스케일러 경로 검증
        model_path = os.path.join(model_dir, 'lstm_model.h5')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        
        missing = []
        if not os.path.exists(model_path): missing.append(model_path)
        if not os.path.exists(scaler_path): missing.append(scaler_path)
        if missing:
            raise FileNotFoundError(f"필수 파일 누락: {', '.join(missing)}")

        # 모델 로드 및 컴파일
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mse')  # 원본 모델 설정과 동일하게
        
        # 스케일러 로드
        scaler = joblib.load(scaler_path)
        if not isinstance(scaler, MinMaxScaler):
            raise TypeError("잘못된 스케일러 타입입니다. MinMaxScaler가 필요합니다.")

        return model, scaler, base_dir

    def select_csv(self):
        """향상된 파일 선택기"""
        files = [f for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
        
        if not files:
            raise FileNotFoundError(f"CSV 디렉토리에 파일이 없습니다: {self.csv_dir}")
        
        print("\n사용 가능한 CSV 파일:")
        for i, f in enumerate(files, 1):
            print(f"[{i}] {f}")
            
        while True:
            try:
                choice = int(input("선택할 파일 번호 입력 (종료: 0): "))
                if choice == 0:
                    raise SystemExit("사용자에 의해 종료됨")
                selected = files[choice-1]
                return os.path.join(self.csv_dir, selected)
            except (ValueError, IndexError):
                print(f"1~{len(files)} 사이의 숫자를 입력해주세요.")

    def prepare_data(self, csv_path):
        """강화된 데이터 검증 파이프라인"""
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='euc-kr')  # 한국어 인코딩 대응

        # 컬럼 검증
        required_cols = ['basDt', 'clpr']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼 누락: {', '.join(missing)}")

        # 데이터 전처리
        df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d', errors='coerce')
        if df['basDt'].isnull().any():
            raise ValueError("유효하지 않은 날짜 형식이 포함되어 있습니다.")
            
        df.sort_values('basDt', inplace=True)
        
        # 시계열 시퀀스 검증
        seq_length = self.model.input_shape[1]
        if len(df) < seq_length + 1:
            required = seq_length + 1
            raise ValueError(f"데이터 부족! 최소 {required}개 데이터 필요 (현재 {len(df)}개)")

        # 데이터 스케일링
        try:
            scaled = self.scaler.transform(df[['clpr']])
        except ValueError as e:
            raise ValueError(f"스케일링 오류: {str(e)}")

        # 시계열 데이터 생성
        X, y, dates = [], [], []
        for i in range(len(scaled)-seq_length):
            X.append(scaled[i:i+seq_length])
            y.append(scaled[i+seq_length])
            dates.append(df['basDt'].iloc[i+seq_length])
            
        return np.array(X), np.array(y), np.array(dates)

    def run_validation(self):
        """개선된 검증 프로세스"""
        try:
            csv_path = self.select_csv()
            X_val, y_val, dates = self.prepare_data(csv_path)
            
            # 예측 수행
            preds = self.model.predict(X_val, verbose=1)
            
            # 스케일 복원
            preds_orig = self.scaler.inverse_transform(preds)
            actual_orig = self.scaler.inverse_transform(y_val.reshape(-1, 1))
            
            # 결과 저장
            result_file = os.path.join(self.results_dir, 'validation_results.csv')
            result_df = pd.DataFrame({
                'Date': dates,
                'Actual': actual_orig.flatten(),
                'Predicted': preds_orig.flatten()
            })
            result_df.to_csv(result_file, index=False)
            
            # 성능 지표 계산
            mse = tf.keras.metrics.mean_squared_error(actual_orig, preds_orig).numpy().mean()
            mae = tf.keras.metrics.mean_absolute_error(actual_orig, preds_orig).numpy().mean()
            accuracy = np.mean(1 - np.abs(actual_orig - preds_orig) / actual_orig) * 100

            print("\n=== 검증 결과 요약 ===")
            print(f"• 평균 제곱 오차(MSE): {mse:,.2f}")
            print(f"• 평균 절대 오차(MAE): {mae:,.2f}")
            print(f"• 예측 정확도: {accuracy:.2f}%")
            print(f"\n결과 파일 경로: {result_file}")
            
            return result_df
            
        except Exception as e:
            raise RuntimeError(f"검증 실패: {str(e)}") from e

if __name__ == "__main__":
    validator = ValidationEngine()
    try:
        results = validator.run_validation()
        print("\n검증 프로세스가 성공적으로 완료되었습니다!")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
    finally:
        input("\n엔터 키를 눌러 종료...")
