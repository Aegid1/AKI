import os
import random
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

        features_to_normalize = ["gdp", "unemployment_rate", "inflation", "interest_rates", "stock_prices", "oil_prices", "currency_rates"]
        data = {key: [] for key in features_to_normalize}

        #at initialization get all scalars
        for file in self.paths:
            with open(os.path.join(path, file), 'rb') as f:
                item = pickle.load(f)
            for key in data:
                data[key].append(item["features"][key][0])

        self.scalers = {key: MinMaxScaler() for key in data}
        print(len(self.scalers))
        #the scalers for the scalars need to be fit during initialization
        for key in data:
            data = np.array(data[key]).reshape(-1, 1)
            self.scalers[key].fit(data)

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
        stock_prices_normalized = self.scalers["stock_prices"].transform(
            np.array(stock_prices).reshape(-1, 1)
        ).flatten()

        oil_prices_normalized = self.scalers["oil_prices"].transform(
            np.array(oil_prices).reshape(-1, 1)
        ).flatten()

        currency_rates_normalized = self.scalers["currency_rates"].fit_transform(
            np.array(currency_rates).reshape(-1, 1)
        ).flatten()

        # normalization of scalars need to be handled differently
        scalars_to_normalize = ["gdp", "unemployment_rate", "inflation", "interest_rates"]
        normalized_scalars = {
            key: self.scalers[key].transform([[item["features"][key][0]]])[0][0]
            for key in scalars_to_normalize
        }

        target = item["target"]

        features = {
            "stock_prices": torch.tensor(stock_prices_normalized, dtype=torch.float32),
            "oil_prices": torch.tensor(oil_prices_normalized, dtype=torch.float32),
            "currency_rates": torch.tensor(currency_rates_normalized, dtype=torch.float32),
            **{
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in normalized_scalars.items()
            },
        }

        return features, torch.tensor(target, dtype=torch.float32)
