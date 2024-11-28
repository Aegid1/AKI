import os
import random

import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pickle

class TechnicalIndicatorsDataSet(Dataset):

    def __init__(self, stocks_seq_size):
        path = os.path.join('..', '..', 'data', 'Samples', 'experiment2')
        self.paths = os.listdir(path)
        self.paths = [p for p in self.paths if p.endswith('.pkl')]
        random.shuffle(self.paths)

        self.stocks_seq_size = stocks_seq_size

        company_names = ["Airbus", "Allianz", "Deutsche Telekom", "Mercedes-Benz", "Volkswagen", "Porsche", "SAP", "Siemens",
         "Siemens Healthineers", "Deutsche Bank"]
        indicators_to_normalize = ["stock_prices", "rsi", "macd", "macdsignal", "macdhist", "sma", "ema", "upperband", "lowerband", "middleband"]
        self.company_indicators_combined =[f"{company}_{indicator}" for company in company_names for indicator in indicators_to_normalize]

        data = {key: [] for key in self.company_indicators_combined}

        # at initialization get all scalars
        for file in self.paths:
            with open(os.path.join(path, file), 'rb') as f:
                item = pickle.load(f)
            for company in company_names:
                if item["company_name"] == company:
                    for indicator in indicators_to_normalize:
                        data[f"{company}_{indicator}"].append(item["features"][indicator][0])

        self.scalers = {key: MinMaxScaler() for key in self.company_indicators_combined}

        # the scalers need to be fit during initialization
        for key in self.company_indicators_combined:
            indicator_data = np.array(data[key]).reshape(-1, 1)
            self.scalers[key].fit(indicator_data)
            scaler_path = os.path.join("scalers", f"{key}_scaler.pkl")
            joblib.dump(self.scalers[key], scaler_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join('..', '..', 'data', 'Samples', 'experiment2')
        with open(os.path.join(path, self.paths[idx]), 'rb') as f:
            item = pickle.load(f)

        indicators_to_normalize = ["rsi", "macd", "macdsignal", "macdhist", "sma", "ema", "upperband", "lowerband", "middleband"]
        company_name = item["company_name"]
        stock_prices = item["features"]["stock_prices"]

        target = item["target"]
        stock_prices_with_target = np.append(stock_prices, target)  # Combine stock prices and target

        stock_prices_normalized_with_target = self.scalers[f"{company_name}_stock_prices"].transform(
            stock_prices_with_target.reshape(-1, 1)
        ).flatten()

        stock_prices_normalized = stock_prices_normalized_with_target[:-1]  # All except the last value
        target_normalized = stock_prices_normalized_with_target[-1]

        normalized_indicators = {
            key: self.scalers[f"{company_name}_{key}"].transform(item["features"][key].reshape(-1, 1))
            for key in indicators_to_normalize
        }

        features = {
            "stock_prices": torch.tensor(stock_prices_normalized, dtype=torch.float32),
            **{
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in normalized_indicators.items()
            },
        }
        return features, torch.tensor(target_normalized, dtype=torch.float32)
