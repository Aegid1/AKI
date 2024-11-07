import torch
import torch.nn as nn
import experiments.Experiment1.network_layers_sentiments as milstm

class Net(nn.Module):

    def __init__(self, seq_size, hidden_size):
        super(Net, self).__init__()
        self.seq_size = seq_size
        self.Stock_price_layer = milstm.CustomLSTM(1, hidden_size)
        self.Short_term_impact_layer = milstm.CustomLSTM(1, hidden_size)
        self.Long_term_impact_layer = milstm.CustomLSTM(1, hidden_size)

        #hidden_size der vorherigen LSTMS bestimmen die Form des Eingabevektors für MI-LSTM
        self.MI_LSTM_layer = milstm.MultiInputLSTMSentiments(hidden_size, hidden_size)
        self.Attention_layer = milstm.Attention(hidden_size, hidden_size)

        self.lin_layer = nn.Linear(hidden_size, 1)

    def forward(self, Stock_price, Short_impact, Long_impact):
        Stock_price_tilde, Stock_price_tilde_hidden = self.Stock_price_layer(Stock_price.unsqueeze(2))
        Short_impact_tilde, Short_impact_tilde_hidden = self.Short_term_impact_layer(Short_impact.unsqueeze(2))
        Long_impact_tilde, Long_impact_tilde_hidden = self.Long_term_impact_layer(Long_impact.unsqueeze(2))

        Y_tilde_prime_out, Y_tilde_prime_hidden = self.MI_LSTM_layer(Stock_price_tilde_hidden, Short_impact_tilde_hidden, Long_impact_tilde_hidden)

        y_tilde = self.Attention_layer(Y_tilde_prime_hidden)
        output = torch.relu(self.lin_layer(y_tilde))
        return output

