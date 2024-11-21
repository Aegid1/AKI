import os
import pickle

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


    def create_stock_pickle_files(self, experiment:str):
        company_names = ["Airbus", "Allianz", "Deutsche Telekom", "Mercedes-Benz", "Volkswagen", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Adidas"]
        input_dir = '/data/stocks/'
        output_dir = f'/data/Samples/{experiment}/'

        for company in company_names:
            company_path = os.path.join(input_dir, company)
            csv_files = sorted([f for f in os.listdir(company) if f.endswith('.csv')])
            all_data = pd.DataFrame()
            for file in csv_files:
                file_path = os.path.join(company_path, file)
                df = pd.read_csv(file_path, parse_dates=['Datetime'])
                all_data = pd.concat([all_data, df])

            all_data.sort_values(by='Datetime', inplace=True)
            all_data.reset_index(drop=True, inplace=True)

            for i in range(len(all_data) - 25):
                sample = all_data.iloc[i:i + 26]
                x = sample['Open'][:-1].values
                y = sample['Open'].iloc[-1]

                item = {'features': x,'target': y}
                pickle_filename = f"{company}_{sample['Datetime'].iloc[0].strftime('%Y%m%d%H%M')}.pkl"
                with open(os.path.join(output_dir, pickle_filename), 'wb') as f:
                    pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
