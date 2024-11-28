import math

import torch
from torch import nn
from experiments.experiment2.network_layers import MultiInputLSTMMacroFactors as MILSTM

class Model(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(10):
            self.layers.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1))

        self.MI_LSTM_layer = MILSTM(hidden_size, hidden_size)

        self.DNN_Layer = nn.Linear(in_features=hidden_size, out_features=1)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    #features as a list [stock_price, rsi, macd, macdsignal, macdhist, sma, ema, upperband, middleband, lowerband]
    def forward(self, features):
        list_tilde = list()
        list_hidden = list()
        for i in range(len(self.layers)):
            layer_out, layer_hidden = self.layers[i](features[i])
            layer_out = torch.relu(layer_out)
            list_tilde.append(layer_out)
            list_hidden.append(layer_hidden)

        milstm_tilde, milst_hidden = self.MI_LSTM_layer(
            list_tilde[0],
            list_tilde[1],
            list_tilde[2],
            list_tilde[3],
            list_tilde[4],
            list_tilde[5],
            list_tilde[6],
            list_tilde[7],
            list_tilde[8],
            list_tilde[9],
        ) #milstm_tilde [50, 25, 200]
        milstm_tilde = torch.relu(milstm_tilde)
        out = self.DNN_Layer(milstm_tilde)
        output = torch.relu(out)
        return output
