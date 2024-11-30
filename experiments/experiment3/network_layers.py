import torch
import torch.nn as nn
import math

class MultiInputLSTMMacroFactors(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        # i_t
        self.W_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # i_p_t
        self.W_i_p = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_p = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_p = nn.Parameter(torch.Tensor(hidden_sz))

        # i_n_t
        self.W_i_n = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_n = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_n = nn.Parameter(torch.Tensor(hidden_sz))

        # f_t
        self.W_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # c_t
        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # c_p_t
        self.W_c_p = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c_p = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c_p = nn.Parameter(torch.Tensor(hidden_sz))

        # c_n_t
        self.W_c_n = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c_n = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c_n = nn.Parameter(torch.Tensor(hidden_sz))

        # o_t
        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        # a_t
        self.W_a = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_a = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, O, C):
        bs, seq_sz, _ = O.shape
        hidden_seq = []
        #initialize start with zeros
        h_t, c_t = (
            torch.zeros(bs, self.hidden_size).to(O.device),
            torch.zeros(bs, self.hidden_size).to(O.device),
        )
        for t in range(seq_sz):
            P_t = O[:, t, :]
            N_t = C[:, t, :]

            #if i have more features or more lstms in the previous step i need more matrices
            i_p_t = torch.relu(P_t @ self.W_i_p + h_t @ self.U_i_p + self.b_i_p) #TODO checken ob hier nicht die gleiche Matrix verwendet werden sollte
            i_n_t = torch.relu(N_t @ self.W_i_n + h_t @ self.U_i_n + self.b_i_n)

            #this always stays the same
            #f_t = torch.sigmoid(P_t @ self.W_f + h_t @ self.U_f + self.b_f) #TODO Dont use the forget gate, maybe test a little with it -> oil or currency

            #if I have more features or more lstms in the previous step I need more matrices
            C_p_tilde_t = torch.relu(P_t @ self.W_c_p + h_t @ self.U_c_p + self.b_c_p)
            C_n_tilde_t = torch.relu(N_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)

            #this always stays the same
            o_t_1 = torch.relu(P_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_2 = torch.relu(N_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t = o_t_1 + o_t_2 #TODO test around if this makes a difference or not -> both features are equally strong

            #if i have more features or more lstms in the previous step i need more values
            l_p_t = C_p_tilde_t * i_p_t
            l_n_t = C_n_tilde_t * i_n_t

            #if i have more features or more lstms in the previous step i need more values, but matrix stays the same
            u_p_t = torch.relu(l_p_t @ self.W_a * c_t + self.b_a)
            u_n_t = torch.relu(l_n_t @ self.W_a * c_t + self.b_a)

            #if i have more features or more lstms in the previous step i need more matrices
            alpha_t = torch.softmax(torch.stack([u_p_t, u_n_t]), dim=0) #TODO check if this is the attention mechanism in the MILSTM
            L_t = alpha_t[0, :, :] * l_p_t + alpha_t[1, :, :] * l_n_t

            #c_t = f_t * c_t + L_t TODO Dont use the forget gate, maybe test a little with it -> oil or currency
            c_t = c_t + L_t
            h_t = o_t * torch.relu(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return h_t, hidden_seq


class Attention(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        # i_t
        self.W_b = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.b_b = nn.Parameter(torch.Tensor(hidden_sz))
        self.v_b = nn.Parameter(torch.Tensor(input_sz, hidden_sz))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, stock_tilde, milstm_tilde, interest_rate, gdp, inflation, unemployment):
        input_vector = torch.stack([stock_tilde, milstm_tilde, interest_rate, gdp, inflation, unemployment], dim=1)
        temp = input_vector @ self.W_b

        j_t = torch.tanh(temp + self.b_b)
        j_t = j_t @ self.v_b.t()
        beta = torch.softmax(j_t, dim=1)
        #print(beta)
        y_tilde = beta * input_vector
        return y_tilde.squeeze()
