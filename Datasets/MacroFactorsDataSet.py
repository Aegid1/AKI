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

        scalars_to_normalize = ["gdp", "unemployment_rate", "inflation", "interest_rates", "stock_prices", "oil_prices", "currency_rates"]
        scalar_data = {key: [] for key in scalars_to_normalize}

        #at initialization get all scalars
        for file in self.paths:
            with open(os.path.join(path, file), 'rb') as f:
                item = pickle.load(f)
            for key in scalars_to_normalize:
                scalar_data[key].append(item["features"][key][0])

        self.scalers = {key: MinMaxScaler() for key in scalars_to_normalize}
        #the scalers for the scalars need to be fit during initialization
        for key in scalars_to_normalize:
            data = np.array(scalar_data[key]).reshape(-1, 1)
            self.scalers[key].fit(data)
            scaler_path = os.path.join("scalers", f"{key}_scaler.pkl")
            joblib.dump(self.scalers[key], scaler_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join('..', '..', 'data', 'Samples', 'experiment3')
        with open(os.path.join(path, self.paths[idx]), 'rb') as f:
            item = pickle.load(f)

        stock_prices = item["features"]["stock_prices"]
        oil_prices = item["features"]["oil_prices"]
        currency_rates = item["features"]["currency_rates"]

        #normalization of sequential data
        target = item["target"]
        stock_prices_with_target = np.append(stock_prices, target)  # Combine stock prices and target
        stock_prices_normalized_with_target = self.scalers["stock_prices"].transform(
            stock_prices_with_target.reshape(-1, 1)
        ).flatten()

        stock_prices_normalized = stock_prices_normalized_with_target[:-1]  # All except the last value
        target_normalized = stock_prices_normalized_with_target[-1]

        oil_prices_normalized = self.scalers["oil_prices"].transform(
            np.array(oil_prices).reshape(-1, 1)
        ).flatten()

        currency_rates_normalized = self.scalers["currency_rates"].transform(
            np.array(currency_rates).reshape(-1, 1)
        ).flatten()

        # normalization of scalars
        scalars_to_normalize = ["gdp", "unemployment_rate", "inflation", "interest_rates"]
        normalized_scalars = {
            key: self.scalers[key].transform([[item["features"][key][0]]])[0][0]
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
