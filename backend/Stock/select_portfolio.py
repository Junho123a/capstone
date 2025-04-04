import os
import pandas as pd
import numpy as np

class VolatilityProcessor:
    def __init__(self):
        """
        초기화 메서드로 기본 디렉토리 설정
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_dir = os.path.abspath(os.path.join(self.base_dir, '..', '..', 'CSV'))
        self.results_dir = os.path.abspath(os.path.join(self.base_dir, '..', '..', 'results'))
        os.makedirs(self.results_dir, exist_ok=True)

    def calculate_volatility_and_trend(self, file_path):
        """
        주어진 CSV 파일에서 변동성과 상승/하락 여부를 계산합니다.
        """
        df = pd.read_csv(file_path, encoding='utf-8')
        df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d', errors='coerce')
        df.sort_values('basDt', inplace=True)
        
        # 종가 데이터(clpr)의 표준편차를 통해 변동성 계산
        volatility = df['clpr'].std()

        # 상승/하락 여부 계산 (첫 번째 종가와 마지막 종가 비교)
        if df['clpr'].iloc[0] < df['clpr'].iloc[-1]:
            trend = "Up"
        else:
            trend = "Down"
        
        return volatility, trend

    def filter_files_by_date(self, start_date, end_date):
        """
        파일 이름에 특정 날짜 범위를 포함하는 CSV 파일 필터링
        """
        csv_files = [os.path.join(self.csv_dir, f) for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
        
        # 날짜 범위로 필터링
        filtered_files = []
        for file in csv_files:
            filename = os.path.basename(file)
            try:
                date_part = filename.split('_')[-2]  # 파일명에서 날짜 추출 (예: '이름_20200102_20231231_KOSPI.csv')
                if start_date <= date_part <= end_date:
                    filtered_files.append(file)
            except IndexError:
                print(f"파일 이름에서 날짜를 추출할 수 없습니다: {filename}")
        
        return filtered_files

    def split_and_save_volatility_groups(self, start_date, end_date):
        """
        변동성이 높은 그룹과 낮은 그룹으로 나누고 상승/하락 여부를 포함하여 하나의 CSV 파일에 저장합니다.
        """
        # 날짜 범위에 해당하는 파일 필터링
        filtered_files = self.filter_files_by_date(start_date, end_date)
        
        if len(filtered_files) < 20:
            raise ValueError("날짜 범위에 해당하는 CSV 파일이 20개 이상 필요합니다.")
        
        # 각 파일의 변동성과 상승/하락 여부 계산
        volatility_data = []
        for file in filtered_files:
            try:
                vol, trend = self.calculate_volatility_and_trend(file)
                volatility_data.append((os.path.basename(file), vol, trend))
            except Exception as e:
                print(f"파일 처리 중 오류 발생: {file}, 오류: {str(e)}")
        
        # 변동성을 기준으로 정렬
        volatility_data.sort(key=lambda x: x[1], reverse=True)  # 높은 변동성 순으로 정렬
        
        # 상위 10개와 하위 10개 그룹 분리
        high_volatility_group = volatility_data[:10]
        low_volatility_group = volatility_data[-10:]
        
        # 하나의 데이터프레임으로 통합
        combined_df = pd.DataFrame({
            'Group': ['High Volatility'] * len(high_volatility_group) + ['Low Volatility'] * len(low_volatility_group),
            'File': [x[0] for x in high_volatility_group + low_volatility_group],
            'Volatility': [x[1] for x in high_volatility_group + low_volatility_group],
            'Trend': [x[2] for x in high_volatility_group + low_volatility_group]
        })
        
        # 저장
        self.save_to_csv(combined_df, "volatility_summary", start_date, end_date, "KOSPI")
        
    def save_to_csv(self, df, item_name, start_date, end_date, market_code):
        """
        데이터를 지정된 디렉토리에 저장합니다.
        """
        item_name_simple = item_name.replace(" ", "_") if item_name else "all_stocks"
        market_str = market_code if market_code else "all_markets"
        
        filename = f"{item_name_simple}_{start_date}_{end_date}_{market_str}.csv"
        
        file_path = os.path.join(self.results_dir, filename)
        
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"Data saved to: {file_path}")

if __name__ == "__main__":
    processor = VolatilityProcessor()
    try:
        # Step 1: 날짜 범위 지정 (예: 20200102 ~ 20231231)
        start_date = "20200102"
        end_date = "20231231"
        
        # Step 2: 변동성 기준으로 그룹 나누기 및 저장
        processor.split_and_save_volatility_groups(start_date, end_date)
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
