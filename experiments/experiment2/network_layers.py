import torch
import torch.nn as nn
import math

class MultiInputLSTMMacroFactors(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        # i_s_t
        self.W_i_s = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_s = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_s = nn.Parameter(torch.Tensor(hidden_sz))

        # i_r_t
        self.W_i_r = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_r = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_r = nn.Parameter(torch.Tensor(hidden_sz))

        # i_m_t
        self.W_i_m = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_m = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_m = nn.Parameter(torch.Tensor(hidden_sz))

        # i_ms_t
        self.W_i_ms = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_ms = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_ms = nn.Parameter(torch.Tensor(hidden_sz))

        # i_mh_t
        self.W_i_mh = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_mh = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_mh = nn.Parameter(torch.Tensor(hidden_sz))

        # i_sm_t
        self.W_i_sm = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_sm = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_sm = nn.Parameter(torch.Tensor(hidden_sz))

        # i_em_t
        self.W_i_em = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_em = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_em = nn.Parameter(torch.Tensor(hidden_sz))

        # i_u_t
        self.W_i_u = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_u = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_u = nn.Parameter(torch.Tensor(hidden_sz))

        # i_mb_t
        self.W_i_mb = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_mb = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_mb = nn.Parameter(torch.Tensor(hidden_sz))

        # i_l_t
        self.W_i_l = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i_l = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i_l = nn.Parameter(torch.Tensor(hidden_sz))

        # f_t
        # self.W_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        # self.U_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        # self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

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

    def forward(self, stock, rsi, macd, macdsignal, macdhist, sma, ema, upperband, middleband, lowerband):
        bs, seq_sz, _ = stock.shape
        hidden_seq = []
        #initialize start with zeros
        h_t, c_t = (
            torch.zeros(bs, self.hidden_size).to(stock.device),
            torch.zeros(bs, self.hidden_size).to(stock.device),
        )
        for t in range(seq_sz):
            S_t = stock[:, t, :]
            R_t = rsi[:, t, :]
            M_t = macd[:, t, :]
            MS_t = macdsignal[:, t, :]
            MH_t = macdhist[:, t, :]
            SM_t = sma[:, t, :]
            EM_t = ema[:, t, :]
            U_t = upperband[:, t, :]
            MB_t = middleband[:, t, :]
            L_t = lowerband[:, t, :]

            #if i have more features or more lstms in the previous step i need more matrices
            i_s_t = torch.relu(S_t @ self.W_i_s + h_t @ self.U_i_s + self.b_i_s) #TODO checken ob hier nicht die gleiche Matrix verwendet werden sollte
            i_r_t = torch.relu(R_t @ self.W_i_r + h_t @ self.U_i_r + self.b_i_r)
            i_m_t = torch.relu(M_t @ self.W_i_m + h_t @ self.U_i_m + self.b_i_m)
            i_ms_t = torch.relu(MS_t @ self.W_i_ms + h_t @ self.U_i_ms + self.b_i_ms)
            i_mh_t = torch.relu(MH_t @ self.W_i_mh + h_t @ self.U_i_mh + self.b_i_mh)
            i_sm_t = torch.relu(SM_t @ self.W_i_sm + h_t @ self.U_i_sm + self.b_i_sm)
            i_em_t = torch.relu(EM_t @ self.W_i_em + h_t @ self.U_i_em + self.b_i_em)
            i_u_t = torch.relu(U_t @ self.W_i_u + h_t @ self.U_i_u + self.b_i_u)
            i_mb_t = torch.relu(MB_t @ self.W_i_mb + h_t @ self.U_i_mb + self.b_i_mb)
            i_l_t = torch.relu(L_t @ self.W_i_l + h_t @ self.U_i_l + self.b_i_l)

            #if I have more features or more lstms in the previous step I need more matrices
            S_tilde_t = torch.relu(S_t @ self.W_c_p + h_t @ self.U_c_p + self.b_c_p)
            R_tilde_t = torch.relu(R_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            M_tilde_t = torch.relu(R_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            MS_tilde_t = torch.relu(R_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            MH_tilde_t = torch.relu(R_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            SM_tilde_t = torch.relu(R_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            EM_tilde_t = torch.relu(R_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            U_tilde_t = torch.relu(R_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            MB_tilde_t = torch.relu(R_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)
            L_tilde_t = torch.relu(R_t @ self.W_c_n + h_t @ self.U_c_n + self.b_c_n)

            #add more outputs
            o_t_1 = torch.relu(S_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_2 = torch.relu(R_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_3 = torch.relu(M_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_4 = torch.relu(MS_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_5 = torch.relu(MH_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_6 = torch.relu(SM_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_7 = torch.relu(EM_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_8 = torch.relu(U_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_9 = torch.relu(MB_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t_10 = torch.relu(L_t @ self.W_o + h_t @ self.U_o + self.b_o)
            o_t = o_t_1 + o_t_2 + o_t_3 + o_t_4 + o_t_5 + o_t_6 + o_t_7 + o_t_8 + o_t_9 +o_t_10

            #if i have more features or more lstms in the previous step i need more values
            l_s_t = S_tilde_t * i_s_t
            l_r_t = R_tilde_t * i_r_t
            l_m_t = M_tilde_t * i_m_t
            l_ms_t = MS_tilde_t * i_ms_t
            l_mh_t = MH_tilde_t * i_mh_t
            l_sm_t = SM_tilde_t * i_sm_t
            l_em_t = EM_tilde_t * i_em_t
            l_u_t = U_tilde_t * i_u_t
            l_mb_t = MB_tilde_t * i_mb_t
            l_l_t = L_tilde_t * i_l_t

            #if i have more features or more lstms in the previous step i need more values, but matrix stays the same
            u_s_t = torch.relu(l_s_t @ self.W_a * c_t + self.b_a)
            u_r_t = torch.relu(l_r_t @ self.W_a * c_t + self.b_a)
            u_m_t = torch.relu(l_m_t @ self.W_a * c_t + self.b_a)
            u_ms_t = torch.relu(l_ms_t @ self.W_a * c_t + self.b_a)
            u_mh_t = torch.relu(l_mh_t @ self.W_a * c_t + self.b_a)
            u_sm_t = torch.relu(l_sm_t @ self.W_a * c_t + self.b_a)
            u_em_t = torch.relu(l_em_t @ self.W_a * c_t + self.b_a)
            u_u_t = torch.relu(l_u_t @ self.W_a * c_t + self.b_a)
            u_mb_t = torch.relu(l_mb_t @ self.W_a * c_t + self.b_a)
            u_l_t = torch.relu(l_l_t @ self.W_a * c_t + self.b_a)

            #if i have more features or more lstms in the previous step i need more matrices
            alpha_t = torch.softmax(torch.stack([u_s_t, u_r_t, u_m_t, u_ms_t, u_mh_t, u_sm_t, u_em_t, u_u_t, u_mb_t, u_l_t]), dim=0)
            L_t = (
                alpha_t[0, :, :] * l_s_t
                + alpha_t[1, :, :] * l_r_t
                + alpha_t[2, :, :] * l_m_t
                + alpha_t[3, :, :] * l_ms_t
                + alpha_t[4, :, :] * l_mh_t
                + alpha_t[5, :, :] * l_sm_t
                + alpha_t[6, :, :] * l_em_t
                + alpha_t[7, :, :] * l_u_t
                + alpha_t[8, :, :] * l_mb_t
                + alpha_t[9, :, :] * l_l_t
            )
            # print("stock_prices Attention Weight: ")
            # print(alpha_t[0, :, :])
            # print("rsi Attention Weight: ")
            # print(alpha_t[1, :, :])
            # print("macd Attention Weight: ")
            # print(alpha_t[2, :, :])
            # print("macdsignal Attention Weight: ")
            # print(alpha_t[3, :, :])
            # print("macdhist Attention Weight: ")
            # print(alpha_t[4, :, :])
            # print("sma Attention Weight: ")
            # print(alpha_t[5, :, :])
            # print("ema Attention Weight: ")
            # print(alpha_t[6, :, :])
            # print("upperband Attention Weight: ")
            # print(alpha_t[7, :, :])
            # print("middleband Attention Weight: ")
            # print(alpha_t[8, :, :])
            # print("lowerband Attention Weight: ")
            # print(alpha_t[9, :, :])

            c_t = c_t + L_t
            h_t = o_t * torch.relu(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return h_t, hidden_seq
