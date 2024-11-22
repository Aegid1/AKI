import math

import torch
from torch import nn
from network_layers import MultiInputLSTMMacroFactors, Attention as MILSTM, Attention


class Model(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.Stock_price_layer = nn.LSTM(input_size=input_size,
                                         hidden_size=hidden_size,
                                         num_layers=1
                                         )
        self.Oil_price_layer = nn.LSTM(input_size=input_size,
                                         hidden_size=hidden_size,
                                         num_layers=1
                                         )
        self.Currency_layer = nn.LSTM(input_size=input_size,
                                         hidden_size=hidden_size,
                                         num_layers=1
                                         )
        self.MI_LSTM_layer = MILSTM(hidden_size, hidden_size)
        self.Attention_layer = Attention(hidden_size, hidden_size)
        self.DNN_Layer = nn.Linear(in_features=hidden_size, out_features=1)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, stock_price, oil_price, currency_rate, interest_rate, gdp, unemployment_rate, inflation_rate):
        stock_price_tilde, stock_price_hidden = self.Stock_price_layer(stock_price)
        oil_price_tilde, oil_price_hidden = self.Oil_price_layer(oil_price)
        currency_rate_tilde, currency_rate_hidden = self.Currency_layer(currency_rate)
        milstm_tilde, milst_hidden = self.MI_LSTM_layer(oil_price_hidden, currency_rate_hidden)
        stock_price_out = stock_price_tilde[:, -1, :]
        attention_out = self.Attention_layer(
            stock_tilde= stock_price_out,
            milstm_tilde= milstm_tilde,
            interest_rate= interest_rate,
            gdp= gdp,
            inflation_rate= inflation_rate,
            unemployment_rate= unemployment_rate
        )
        out = self.DNN_Layer(attention_out)
        output = torch.relu(out)
        return output
