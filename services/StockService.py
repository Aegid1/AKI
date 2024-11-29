import os
import pandas as pd
import yfinance as yf

class StockService:

    def save_stock_data(self, ticker_symbol:str, start_date:str, end_date:str, company_name:str):
        """
            Downloads hourly stock data for a given ticker symbol and saves it as weekly CSV files.

            This function uses the Yahoo Finance API to download stock data for a specific ticker symbol between the
            specified start and end dates. It then filters the data to include only specific times of the day (09:00, 11:00,
            13:00, 15:00, and 17:00), adds year and week information, and saves the data in CSV format for each week.

            Parameters:
            - ticker_symbol (str): The stock ticker symbol (e.g., "AAPL" for Apple).
            - start_date (str): The start date for the stock data (format: "YYYY-MM-DD").
            - end_date (str): The end date for the stock data (format: "YYYY-MM-DD").
            - company_name (str): The name of the company (used to organize the saved files).
        """
        data = yf.download(
            ticker_symbol,
            start=start_date,
            end=end_date,
            interval="1h"
        )
        data.index = data.index.tz_convert("Europe/Berlin")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        desired_times = ["09:00:00", "11:00:00", "13:00:00", "15:00:00", "17:00:00"]
        filtered_data = data[data.index.strftime('%H:%M:%S').isin(desired_times)]

        filtered_data['Year'] = filtered_data.index.year
        filtered_data['Week'] = filtered_data.index.isocalendar().week
        filtered_data['Datetime'] = filtered_data.index

        final_data = filtered_data[['Datetime', 'Open', 'Year', 'Week']]

        save_path = f"data/stocks/{company_name}"

        for (year, week), weekly_data in final_data.groupby(['Year', 'Week']):
            filename = f"{company_name}_{year}_W{week}.csv"
            file_path = os.path.join(save_path, filename)

            weekly_data.to_csv(file_path, index=False)
            print(f"Gespeichert: {file_path}")
