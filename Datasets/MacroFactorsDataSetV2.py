import os
import random

import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pickle

class StocksDataSet(Dataset):

    def __init__(self, stocks_seq_size, oil_seq_size, currency_seq_size):
        path = os.path.join('..', '..', 'data', 'Samples', 'experiment3')
        self.paths = os.listdir(path)
        self.paths = [p for p in self.paths if p.endswith('.pkl')]
        random.shuffle(self.paths)

        self.stocks_seq_size = stocks_seq_size
        self.oil_seq_size = oil_seq_size
        self.currency_seq_size = currency_seq_size

        company_names = ["Airbus", "Allianz", "Deutsche Telekom", "Mercedes-Benz", "Volkswagen", "Porsche", "SAP", "Siemens",
         "Siemens Healthineers", "Deutsche Bank"]

        scalars_to_normalize = ["gdp", "unemployment_rate", "inflation", "interest_rates", "oil_prices", "currency_rates"]
        scalar_data = {key: [] for key in scalars_to_normalize}
        stock_price_data = {company: [] for company in company_names}

        # at initialization get all scalars
        for file in self.paths:
            with open(os.path.join(path, file), 'rb') as f:
                item = pickle.load(f)
            for key in scalars_to_normalize:
                scalar_data[key].append(item["features"][key][0])
            #we want to have the stock prices separated from the other features - due to different normalization strategy
            for company in company_names:
                if item["company_name"] == company: stock_price_data[company].append(item["features"]["stock_prices"][0])

        self.scalar_scalers = {key: MinMaxScaler() for key in scalars_to_normalize}
        self.stock_price_scalers = {key: MinMaxScaler() for key in company_names}

        # the scalers for the scalars need to be fit during initialization
        for key in scalars_to_normalize:
            data = np.array(scalar_data[key]).reshape(-1, 1)
            self.scalar_scalers[key].fit(data)
            scaler_path = os.path.join("scalers", f"{key}_scaler.pkl")
            print(scaler_path)
            joblib.dump(self.scalar_scalers[key], scaler_path)

        for company in company_names:
            data = np.array(stock_price_data[company]).reshape(-1, 1)
            self.stock_price_scalers[company].fit(data)


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join('..', '..', 'data', 'Samples', 'experiment3')
        with open(os.path.join(path, self.paths[idx]), 'rb') as f:
            item = pickle.load(f)

        company_name = item["company_name"]
        stock_prices = item["features"]["stock_prices"]
        oil_prices = item["features"]["oil_prices"]
        currency_rates = item["features"]["currency_rates"]

        target = item["target"]
        stock_prices_with_target = np.append(stock_prices, target)  # Combine stock prices and target
        #normalization of sequential data

        stock_prices_normalized_with_target = self.stock_price_scalers[company_name].transform(
            stock_prices_with_target.reshape(-1, 1)
        ).flatten()

        stock_prices_normalized = stock_prices_normalized_with_target[:-1]  # All except the last value
        target_normalized = stock_prices_normalized_with_target[-1]

        oil_prices_normalized = self.scalar_scalers["oil_prices"].transform(
            np.array(oil_prices).reshape(-1, 1)
        ).flatten()

        currency_rates_normalized = self.scalar_scalers["currency_rates"].transform(
            np.array(currency_rates).reshape(-1, 1)
        ).flatten()

        # normalization of scalars need to be handled differently
        scalars_to_normalize = ["gdp", "unemployment_rate", "inflation", "interest_rates"]
        normalized_scalars = {
            key: self.scalar_scalers[key].transform([[item["features"][key][0]]])[0][0]
            for key in scalars_to_normalize
        }

        features = {
            "stock_prices": torch.tensor(stock_prices_normalized, dtype=torch.float32),
            "oil_prices": torch.tensor(oil_prices_normalized, dtype=torch.float32),
            "currency_rates": torch.tensor(currency_rates_normalized, dtype=torch.float32),
            **{
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in normalized_scalars.items()
            },
        }

        return features, torch.tensor(target_normalized, dtype=torch.float32)
