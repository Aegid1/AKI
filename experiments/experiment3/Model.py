import math

import torch
from torch import nn
from network_layers import MultiInputLSTMMacroFactors as MILSTM
from network_layers import Attention

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
        self.Stock_DNN = nn.Linear(in_features=hidden_size, out_features=1)
        self.MILSTM_DNN = nn.Linear(in_features=hidden_size, out_features=1)

        #input_size is 6 because we use 6 macro factors in this attention layer
        self.Attention_layer = Attention(6, hidden_size)
        self.DNN_Layer = nn.Linear(in_features=6, out_features=1)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, stock_price, oil_price, currency_rate, interest_rate, gdp, unemployment_rate, inflation_rate):
        stock_price_tilde, stock_price_hidden = self.Stock_price_layer(stock_price) #stock_price_tilde [50,25,200]
        oil_price_tilde, oil_price_hidden = self.Oil_price_layer(oil_price) #oil_price_tilde [50,5,200]
        currency_rate_tilde, currency_rate_hidden = self.Currency_layer(currency_rate) #currency_rate_tilde [50,5,200]
        milstm_tilde, milst_hidden = self.MI_LSTM_layer(oil_price_tilde, currency_rate_tilde) #milstm_tilde [50,200]

        stock_price_out = stock_price_tilde[:, -1, :]
        milstm_dnn_output = self.MILSTM_DNN(milstm_tilde).squeeze() #this layer serves as a transformation of the hidden_states to a single scalar
        stock_dnn_output = self.Stock_DNN(stock_price_out).squeeze() #this layer serves as a transformation of the hidden_states to a single scalar
        #attention layer needs all inputs to be in the shape of [50]
        attention_out = self.Attention_layer(
            stock_tilde= stock_dnn_output,
            milstm_tilde= milstm_dnn_output,
            interest_rate= interest_rate,
            gdp= gdp,
            inflation= inflation_rate,
            unemployment= unemployment_rate
        )
        #print(attention_out)
        out = self.DNN_Layer(attention_out)
        output = torch.relu(out)
        return output
