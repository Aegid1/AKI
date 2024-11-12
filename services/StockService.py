import os

import pandas as pd
import yfinance as yf

class StockService:

    def save_stock_data(self, ticker_symbol:str, start_date:str, end_date:str, company_name:str):
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

