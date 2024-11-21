import os
import pickle

import numpy as np
import pandas as pd
from IPython.display import display


def get_last_n_valid_values(series, n=5):
    unique_values = series.dropna().drop_duplicates().values
    first_n_unique_values = unique_values[:n]
    if len(first_n_unique_values) < n:
        first_n_unique_values = np.pad(first_n_unique_values, (0, n - len(first_n_unique_values)), constant_values=first_n_unique_values[-1])
    return first_n_unique_values

def synchronize_data(base_times, oil_prices, currency_rates, gdp, inflation, interest, unemployment_rate):
    base_times = pd.DataFrame({"Datetime": base_times}).sort_values(by="Datetime")
    oil_prices = oil_prices.sort_values(by="Date")
    currency_rates = currency_rates.sort_values(by="Date")
    gdp = gdp.sort_values(by="Date")
    inflation = inflation.sort_values(by="Date")
    interest = interest.sort_values(by="Date")
    unemployment_rate = unemployment_rate.sort_values(by="Date")

    gdp_features = pd.merge_asof(base_times, gdp, left_on="Datetime", right_on="Date", direction="backward")["Value"]
    inflation_features = pd.merge_asof(base_times, inflation, left_on="Datetime", right_on="Date", direction="backward")["Value"]
    interest_features = pd.merge_asof(base_times, interest, left_on="Datetime", right_on="Date", direction="backward")["Value"]
    unemployment_features = pd.merge_asof(base_times, unemployment_rate, left_on="Datetime", right_on="Date", direction="backward")["Value"]

    oil_features = pd.merge_asof(base_times, oil_prices, left_on="Datetime", right_on="Date", direction="backward")["Value"]
    currency_features = pd.merge_asof(base_times, currency_rates, left_on="Datetime", right_on="Date", direction="backward")["Value"]
    oil_features = get_last_n_valid_values(oil_features, n=5)
    currency_features = get_last_n_valid_values(currency_features, n=5)
    gdp_features = get_last_n_valid_values(gdp_features, n=1)
    inflation_features = get_last_n_valid_values(inflation_features, n=1)
    interest_features = get_last_n_valid_values(interest_features, n=1)
    unemployment_features = get_last_n_valid_values(unemployment_features, n=1)
    return oil_features, currency_features, gdp_features, inflation_features, interest_features, unemployment_features


def create_macro_factors_samples():
    company_names = ["Airbus", "Allianz", "Deutsche Telekom", "Mercedes-Benz", "Volkswagen", "Porsche", "SAP",
                     "Siemens", "Siemens Healthineers", "Adidas"]
    input_dir_stocks = '../data/stocks/'
    output_dir = '../data/Samples/experiment3/'
    os.makedirs(output_dir, exist_ok=True)

    oil_prices = pd.read_csv('../data/macro_factors/oil.csv', parse_dates=['Date'])
    oil_prices['Date'] = oil_prices['Date'].dt.tz_localize(None)

    currency_rates = pd.read_csv('../data/macro_factors/currency_euro_dollar.csv', parse_dates=['Date'])
    currency_rates['Date'] = currency_rates['Date'].dt.tz_localize(None)

    gdp = pd.read_csv('../data/macro_factors/gdp.csv', parse_dates=['Date'])
    gdp['Date'] = gdp['Date'].dt.tz_localize(None)

    inflation = pd.read_csv('../data/macro_factors/inflation_rate.csv', parse_dates=['Date'])
    inflation['Date'] = inflation['Date'].dt.tz_localize(None)

    interest = pd.read_csv('../data/macro_factors/central_interest_rate.csv', parse_dates=['Date'])
    interest['Date'] = interest['Date'].dt.tz_localize(None)

    unemployment_rate = pd.read_csv('../data/macro_factors/unemployment_rate.csv', parse_dates=['Date'])
    unemployment_rate['Date'] = unemployment_rate['Date'].dt.tz_localize(None)

    for company in company_names:
        company_path = os.path.join(input_dir_stocks, company)
        csv_files = sorted([f for f in os.listdir(company_path) if f.endswith('.csv')])
        all_data = pd.DataFrame()
        for file in csv_files:
            file_path = os.path.join(company_path, file)
            df = pd.read_csv(file_path, parse_dates=['Datetime'])
            df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
            all_data = pd.concat([all_data, df])

        all_data.sort_values(by='Datetime', inplace=True)
        all_data.reset_index(drop=True, inplace=True)

        for i in range(0, len(all_data), 25):
            if i + 25 <= len(all_data):
                sample = all_data.iloc[i:i + 25]
                stock_features = sample['Open'][:-1].values
                stock_target = sample['Open'].iloc[-1]
                datetime_start = sample['Datetime'].iloc[0]

                # Sicherstellen, dass Datetime korrekt ist
                base_times = pd.to_datetime(sample['Datetime'])  # Konvertiere zu datetime64
                oil_features, currency_features, gdp_features, inflation_features, interest_features, unemployment_features = synchronize_data(base_times, oil_prices, currency_rates, gdp, inflation, interest, unemployment_rate)

                # Sample speichern
                item = {
                    'features': {
                        'stock_prices': stock_features,
                        'oil_prices': oil_features,
                        'currency_rates': currency_features,
                        'gdp': gdp_features,
                        'inflation': inflation_features,
                        'interest_rates': interest_features,
                        'unemployment_rate': unemployment_features,
                    },
                    'target': stock_target,
                    'datetime': datetime_start
                }

                output_filename = os.path.join(output_dir, f'{company}_{datetime_start.strftime("%Y-%m-%d_%H-%M-%S")}.pkl')
                with open(output_filename, 'wb') as f:
                    pickle.dump(item, f)
                print(f"Sample saved: {output_filename}")


create_macro_factors_samples()