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

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join('..', '..', 'data', 'Samples', 'experiment2')
        with open(os.path.join(path, self.paths[idx]), 'rb') as f:
            item = pickle.load(f)

        feature_tensor = torch.tensor(item["features"]["stock_prices"], dtype=torch.float32)
        features = []
        for feature_name, feature_value in item['features'].items():
            if feature_name == "stock_prices":
                stock_prices = item["features"]["stock_prices"]
                features.append(torch.tensor(stock_prices, dtype=torch.float32))
            else:
                feature = item["features"][feature_name]
                features.append(torch.tensor(feature, dtype=torch.float32))


        return torch.tensor(np.array(features), dtype=torch.float32), item["target"]
