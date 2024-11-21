import math

import torch
from torch import nn


class Model(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.Stock_price_layer = nn.LSTM(input_size=input_size,
                                         hidden_size=hidden_size,
                                         num_layers=1
                                         )
        self.DNN_Layer = nn.Linear(in_features=hidden_size, out_features=100)
        self.DNN_Layer2 = nn.Linear(in_features=100, out_features=1)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, stock_price):
        stock_price_tilde, stock_price_tilde_hidden = self.Stock_price_layer(stock_price)
        out = stock_price_tilde[:, -1, :]
        out = self.DNN_Layer(out)
        out = self.DNN_Layer2(out)
        output = torch.relu(out)
        return output
