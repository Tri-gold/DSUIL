import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy
import numpy as np

class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self):
        pass

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        # self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        # torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)

class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., dataset=None, gamma = 0):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.dataset = dataset
        self.gamma = gamma
        self.time_encoder = TimeEncode(expand_dim=self.dim)
        self.W1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_4 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_5 = nn.Parameter(torch.Tensor(self.dim, 1))

        self.time_pro = nn.Parameter(torch.Tensor(self.dim, 1))

        self.b_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.b_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.b_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.b_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.b_4 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.b_5 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.b_6 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.b_7 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.b_8 = nn.Parameter(torch.Tensor(self.dim, 1))
        if self.dataset == 'UB':
            self.b_9 = nn.Parameter(torch.Tensor(self.dim, 1))
            self.b_10 = nn.Parameter(torch.Tensor(self.dim, 1))
            self.b_11 = nn.Parameter(torch.Tensor(self.dim, 1))
            self.b_12 = nn.Parameter(torch.Tensor(self.dim, 1))
            self.b_13 = nn.Parameter(torch.Tensor(self.dim, 1))
            self.b_14 = nn.Parameter(torch.Tensor(self.dim, 1))
            self.b_15 = nn.Parameter(torch.Tensor(self.dim, 1))




        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, beh_adj):
        #print(beh_adj[0])
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)
        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)


        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)


        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha =  torch.where(adj.eq(3), e_2, alpha)


        b_0 = torch.matmul(a_input, self.b_0)
        b_1 = torch.matmul(a_input, self.b_1)
        b_2 = torch.matmul(a_input, self.b_2)
        b_3 = torch.matmul(a_input, self.b_3)
        b_4 = torch.matmul(a_input, self.b_4)
        b_5 = torch.matmul(a_input, self.b_5)
        b_6 = torch.matmul(a_input, self.b_6)
        b_7 = torch.matmul(a_input, self.b_7)
        b_8 = torch.matmul(a_input, self.b_8)
        if self.dataset == "UB":
            b_9 = torch.matmul(a_input, self.b_9)
            b_10 = torch.matmul(a_input, self.b_10)
            b_11 = torch.matmul(a_input, self.b_11)
            b_12 = torch.matmul(a_input, self.b_12)
            b_13 = torch.matmul(a_input, self.b_13)
            b_14 = torch.matmul(a_input, self.b_14)
            b_15 = torch.matmul(a_input, self.b_15)
        b_0 = self.leakyrelu(b_0).squeeze(-1).view(batch_size, N, N)
        b_1 = self.leakyrelu(b_1).squeeze(-1).view(batch_size, N, N)
        b_2 = self.leakyrelu(b_2).squeeze(-1).view(batch_size, N, N)
        b_3 = self.leakyrelu(b_3).squeeze(-1).view(batch_size, N, N)
        b_4 = self.leakyrelu(b_4).squeeze(-1).view(batch_size, N, N)
        b_5 = self.leakyrelu(b_5).squeeze(-1).view(batch_size, N, N)
        b_6 = self.leakyrelu(b_6).squeeze(-1).view(batch_size, N, N)
        b_7 = self.leakyrelu(b_7).squeeze(-1).view(batch_size, N, N)
        b_8 = self.leakyrelu(b_8).squeeze(-1).view(batch_size, N, N)
        if self.dataset == "UB":
            b_9 = self.leakyrelu(b_9).squeeze(-1).view(batch_size, N, N)
            b_10 = self.leakyrelu(b_10).squeeze(-1).view(batch_size, N, N)
            b_11 = self.leakyrelu(b_11).squeeze(-1).view(batch_size, N, N)
            b_12 = self.leakyrelu(b_12).squeeze(-1).view(batch_size, N, N)
            b_13 = self.leakyrelu(b_13).squeeze(-1).view(batch_size, N, N)
            b_14 = self.leakyrelu(b_14).squeeze(-1).view(batch_size, N, N)
            b_15 = self.leakyrelu(b_15).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(b_0)
        alpha_beh = torch.where(beh_adj.eq(1), b_0, mask)
        alpha_beh = torch.where(beh_adj.eq(2), b_1, alpha_beh)
        alpha_beh = torch.where(beh_adj.eq(3), b_2, alpha_beh)
        alpha_beh = torch.where(beh_adj.eq(4), b_3, alpha_beh)
        alpha_beh = torch.where(beh_adj.eq(5), b_4, alpha_beh)
        alpha_beh = torch.where(beh_adj.eq(6), b_5, alpha_beh)
        alpha_beh = torch.where(beh_adj.eq(7), b_6, alpha_beh)
        alpha_beh = torch.where(beh_adj.eq(8), b_7, alpha_beh)
        alpha_beh = torch.where(beh_adj.eq(9), b_8, alpha_beh)
        if self.dataset == "UB":
            alpha_beh = torch.where(beh_adj.eq(10), b_9, alpha_beh)
            alpha_beh = torch.where(beh_adj.eq(11), b_10, alpha_beh)
            alpha_beh = torch.where(beh_adj.eq(12), b_11, alpha_beh)
            alpha_beh = torch.where(beh_adj.eq(13), b_12, alpha_beh)
            alpha_beh = torch.where(beh_adj.eq(14), b_13, alpha_beh)
            alpha_beh = torch.where(beh_adj.eq(15), b_14, alpha_beh)
            alpha_beh = torch.where(beh_adj.eq(16), b_15, alpha_beh)
        alpha_beh = torch.softmax(alpha_beh, dim=-1)
        
        #alpha = torch.softmax(alpha_beh + alpha, dim=-1)
        alpha = torch.softmax(alpha, dim=-1)
        output = torch.matmul(alpha, h)
        output = F.dropout(output, self.dropout, training=self.training)
        output_beh = torch.matmul(alpha_beh, h)
        output_beh = F.dropout(output_beh, self.dropout, training=self.training)
        output = self.gamma * output + output_beh * (1-self.gamma)
        return output

