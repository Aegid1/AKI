import os

import pandas as pd
import talib

class TechnicalindicatorsService:

    def calculate_rsi(self, company_name:str, time_period:int):
        """
            Calculates the Relative Strength Index (RSI) for the given company stock data.

            Parameters:
            - company_name (str): The name of the company (used to locate the stock data).
            - time_period (int): The time period for RSI calculation (e.g., 14).

            Returns:
            - rsi (numpy.ndarray): RSI values for the stock data.
        """
        file_path = os.path.join("data", "stocks", f"{company_name}")
        df = self.get_stock_data_as_df(file_path)

        rsi = talib.RSI(df['Open'], timeperiod=time_period) #time_period = 14
        return rsi


    def calculate_moving_average_convergence_divergence(self, company_name:str, fast_period:int, slow_period:int, signalperiod:int):
        """
            Calculates the Moving Average Convergence Divergence (MACD) for the given company stock data.

            Parameters:
            - company_name (str): The name of the company (used to locate the stock data).
            - fast_period (int): Fast period for MACD (e.g., 12).
            - slow_period (int): Slow period for MACD (e.g., 26).
            - signalperiod (int): Signal period for MACD (e.g., 9).

            Returns:
            - macd (numpy.ndarray): MACD values.
            - macdsignal (numpy.ndarray): MACD signal line values.
            - macdhist (numpy.ndarray): MACD histogram values.
        """
        file_path = os.path.join("data", "stocks", f"{company_name}")
        df = self.get_stock_data_as_df(file_path)

        macd, macdsignal, macdhist = talib.MACD(df['Open'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signalperiod)#14, 26, 9
        return macd, macdsignal, macdhist



    def calculate_simple_moving_average(self, company_name:str, time_period:int):
        """
            Calculates the Simple Moving Average (SMA) for the given company stock data.

            Parameters:
            - company_name (str): The name of the company (used to locate the stock data).
            - time_period (int): The time period for SMA calculation (e.g., 20).

            Returns:
            - sma (numpy.ndarray): SMA values for the stock data.
        """
        file_path = os.path.join("data", "stocks", f"{company_name}")
        df = self.get_stock_data_as_df(file_path)

        sma = talib.SMA(df['Open'], timeperiod=time_period) #time_period = 20
        return sma



    def calculate_exponential_moving_average(self, company_name:str, time_period:int):
        """
            Calculates the Exponential Moving Average (EMA) for the given company stock data.

            Parameters:
            - company_name (str): The name of the company (used to locate the stock data).
            - time_period (int): The time period for EMA calculation (e.g., 20).

            Returns:
            - ema (numpy.ndarray): EMA values for the stock data.
        """
        file_path = os.path.join("data", "stocks", f"{company_name}")
        df = self.get_stock_data_as_df(file_path)

        ema = talib.EMA(df['Open'], timeperiod=time_period) #time_period = 20
        return ema



    def calculate_bollinger_baender(self, company_name: str, time_period: int):
        """
            Calculates the Bollinger Bands for the given company stock data.

            Parameters:
            - company_name (str): The name of the company (used to locate the stock data).
            - time_period (int): The time period for Bollinger Bands calculation (e.g., 20).

            Returns:
            - upperband (numpy.ndarray): Upper Bollinger Band values.
            - middleband (numpy.ndarray): Middle Bollinger Band values (SMA).
            - lowerband (numpy.ndarray): Lower Bollinger Band values.
        """
        file_path = os.path.join("data", "stocks", f"{company_name}")
        df = self.get_stock_data_as_df(file_path)

        upperband, middleband, lowerband = talib.BBANDS(df['Open'], timeperiod=time_period, nbdevup=2, nbdevdn=2) #20
        return upperband, middleband, lowerband



    def get_stock_data_as_df(self, file_path:str):
        """
            Reads and combines CSV files from a directory into a single DataFrame.

            Parameters:
            - file_path (str): Path to the directory containing the CSV files.

            Returns:
            - combined_df (pandas.DataFrame): Combined DataFrame with stock data from all CSV files.
        """
        combined_df = pd.DataFrame()
        for file in os.listdir(file_path):
            if file.endswith(".csv"):
                file_full_path = os.path.join(file_path, file)
                stock_data = pd.read_csv(file_full_path)
                combined_df = pd.concat([combined_df, stock_data], ignore_index=True)
        return combined_df