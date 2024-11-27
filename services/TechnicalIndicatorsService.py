import os

import pandas as pd
import talib

class TechnicalindicatorsService:

    def calculate_rsi(self, company_name:str, time_period:int):
        file_path = os.path.join("data", "stocks", f"{company_name}")
        df = self.get_stock_data_as_df(file_path)

        rsi = talib.RSI(df['Open'], timeperiod=time_period) #time_period = 14
        return rsi


    def calculate_moving_average_convergence_divergence(self, company_name:str, fast_period:int, slow_period:int, signalperiod:int):
        file_path = os.path.join("data", "stocks", f"{company_name}")
        df = self.get_stock_data_as_df(file_path)

        macd, macdsignal, macdhist = talib.MACD(df['Open'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signalperiod)#14, 26, 9
        return macd, macdsignal, macdhist



    def calculate_simple_moving_average(self, company_name:str, time_period:int):
        file_path = os.path.join("data", "stocks", f"{company_name}")
        df = self.get_stock_data_as_df(file_path)

        sma = talib.SMA(df['Open'], timeperiod=time_period) #time_period = 20
        return sma



    def calculate_exponential_moving_average(self, company_name:str, time_period:int):
        file_path = os.path.join("data", "stocks", f"{company_name}")
        df = self.get_stock_data_as_df(file_path)

        ema = talib.EMA(df['Open'], timeperiod=time_period) #time_period = 20
        return ema



    def calculate_bollinger_baender(self, company_name: str, time_period: int):
        file_path = os.path.join("data", "stocks", f"{company_name}")
        df = self.get_stock_data_as_df(file_path)

        upperband, middleband, lowerband = talib.BBANDS(df['Open'], timeperiod=time_period, nbdevup=2, nbdevdn=2) #20
        return upperband, middleband, lowerband



    def get_stock_data_as_df(self, file_path:str):
        combined_df = pd.DataFrame()
        for file in os.listdir(file_path):
            if file.endswith(".csv"):
                file_full_path = os.path.join(file_path, file)
                stock_data = pd.read_csv(file_full_path)
                combined_df = pd.concat([combined_df, stock_data], ignore_index=True)
        return combined_df