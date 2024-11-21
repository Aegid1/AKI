import torch
import torch.nn as nn
import math


class MultiInputLSTMSentiments(nn.Module):
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
        torch.nn.init.zeros_(self.W_c_p)
        torch.nn.init.zeros_(self.U_c_p)
        torch.nn.init.zeros_(self.b_c_p)
        torch.nn.init.zeros_(self.W_c_n)
        torch.nn.init.zeros_(self.U_c_n)
        torch.nn.init.zeros_(self.b_c_n)

    def forward(self, Y, P, N):

        bs, seq_sz, _ = Y.size()
        hidden_seq = []
        h_t, c_t = (
            torch.zeros(bs, self.hidden_size).to(Y.device),
            torch.zeros(bs, self.hidden_size).to(Y.device),
        )

        for t in range(seq_sz):
            Y_t = Y[:, t, :]
            P_t = P[:, t, :]
            N_t = N[:, t, :]

            #if i have more features or more lstms in the previous step i need more matrices
            i_t = torch.sigmoid(Y_t @ self.W_i + h_t @ self.U_i + self.b_i)
            i_p_t = torch.sigmoid(P_t @ self.W_i_p + h_t @ self.U_i_p + self.b_i_p)
            i_n_t = torch.sigmoid(N_t @ self.W_i_n + h_t @ self.U_i_n + self.b_i_n)

            #this always stays the same
            f_t = torch.sigmoid(Y_t @ self.W_f + h_t @ self.U_f + self.b_f)

            #if i have more features or more lstms in the previous step i need more matrices
            C_tilde_t = torch.tanh(Y_t @ self.W_c + h_t @ self.U_c + self.b_c)
            C_p_tilde_t = torch.tanh(P_t @ self.W_c_p + h_t @ self.U_c_p + self.b_c_p)
            C_n_tilde_t = torch.tanh(N_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)

            #this always stays the same
            o_t = torch.sigmoid(Y_t @ self.W_o + h_t @ self.U_o + self.b_o)

            #if i have more features or more lstms in the previous step i need more values
            l_t = C_tilde_t * i_t
            l_p_t = C_p_tilde_t * i_p_t
            l_n_t = C_n_tilde_t * i_n_t

            #if i have more features or more lstms in the previous step i need more values, but matrix stays the same
            u_t = torch.tanh(l_t @ self.W_a * c_t + self.b_a)
            u_p_t = torch.tanh(l_p_t @ self.W_a * c_t + self.b_a)
            u_n_t = torch.tanh(l_n_t @ self.W_a * c_t + self.b_a)

            #if i have more features or more lstms in the previous step i need more matrices
            alpha_t = torch.softmax(torch.stack([u_t, u_p_t, u_n_t]), dim=0)
            L_t = alpha_t[0, :, :] * l_t + alpha_t[1, :, :] * l_p_t + alpha_t[2, :, :] * l_n_t

            c_t = f_t * c_t + L_t
            h_t = o_t * torch.tanh(c_t)

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
        self.v_b = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, Y_tilde_prime):

        j_t_list = list()
        for i in range(20):
            temp = Y_tilde_prime[:, i, :] @ self.W_b
            j_t_list.append(torch.tanh(temp + self.b_b) @ self.v_b.t())

        beta = torch.softmax(torch.stack(j_t_list), dim=0)
        y_tilde = Y_tilde_prime.permute(0,2,1) @ torch.unsqueeze(beta.transpose(0,1), dim=2)
        return y_tilde.squeeze()


class CustomLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))

        self.W_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.W_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.W_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.W_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        bs, seq_sz, _ = x.shape  #assumes x.shape represents (batch_size, sequence_size, input_size)
        hidden_seq = []
        h_t = torch.zeros(bs, self.hidden_size)  #initialize states für den ersten step, da wir hier noch keinen vorherigen Schritt haben
        c_t = torch.zeros(bs, self.hidden_size)

        for t in range(seq_sz):
            x_t = x[:, t, :]

            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.W_f + self.b_f)
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.W_i + self.b_i)
            C_t = torch.tanh(x_t @ self.U_c + h_t @ self.W_c + self.b_c)
            c_t = f_t * c_t + i_t * C_t
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.W_o + self.b_o)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(1))  # transform h_t from shape (batch_size, hidden_size) to shape (batch_size, 1, hidden_size)

        # reshape hidden_seq
        hidden_seq = torch.cat(hidden_seq, dim=1)  # concatenate list of tensors into one tensor along dimension 1 (batch_size, sequence_size, hidden_size)
        return (h_t, c_t), hidden_seq