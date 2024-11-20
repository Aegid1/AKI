import os
import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pickle

class StocksDataSet(Dataset):

    def __init__(self, seq_size):
        self.seq_size = seq_size
        self.paths = os.listdir('../../data/Samples/experiment0')
        self.paths = [p for p in self.paths if p.endswith('.pkl')]
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open('../../data/Samples/experiment0/' + self.paths[idx], 'rb') as f:
            item = pickle.load(f)

        features = np.array(item['features']).reshape(-1, 1)
        target = np.array(item['target']).reshape(-1, 1)
        combined = np.vstack((features, target))

        scaler = MinMaxScaler()
        combined_scaled = scaler.fit_transform(combined)

        features_scaled = combined_scaled[:-1].flatten()
        target_scaled = combined_scaled[-1].flatten()
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        target_tensor = torch.tensor(target_scaled, dtype=torch.float32)
        target_tensor = target_tensor.squeeze()

        if len(features_tensor) < self.seq_size:
            features_tensor = torch.cat((torch.zeros(self.seq_size - len(features_tensor)), features_tensor))
        else:
            features_tensor = features_tensor[-self.seq_size:]

        return features_tensor, target_tensor
