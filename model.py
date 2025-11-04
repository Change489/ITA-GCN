from typing import final

import torch
from scipy.io.arff.setup import configuration
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv


class SelfExpr(nn.Module):
    def __init__(self, n):
        self.n = n
        super(SelfExpr, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(n, n).uniform_(0, 0.01))

    def forward(self, input):
        output = torch.mm(self.weight - torch.diag(torch.diagonal(self.weight)), input)
        return self.weight, output

    def reset(self, input):
        self.weight.data = torch.FloatTensor(self.n, self.n).uniform_(0, 0.01)


class CommunityModel(nn.Module):
    def __init__(self, n_hid1, n_hid2, n_class, dropout):
        super(CommunityModel, self).__init__()
        self.mlp1 = nn.Linear(n_hid1, n_hid2)
        self.mlp2 = nn.Linear(n_hid2, n_class)
        self.dropout = dropout

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        x2 = F.relu(self.mlp1(x1))
        if self.dropout > 0:
            x2 = F.dropout(x2, self.dropout, training=self.training)
        z = F.softmax(self.mlp2(x2), dim=-1)
        return z


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = DenseGCNConv(in_channels, hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, out_channels)

    def forward(self, x, adj):
        x1 = self.conv1(x, adj)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, adj)
        x2 = F.relu(x2)

        final_x = x2.sum(dim=2)

        return x2, final_x


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.gcn1 = GCN(116, 64, 32)
        self.gcn2 = GCN(32, 16, 8)
        self.mlp = MLP(116+config['n_class'],32,2)

    def forward(self, data):
        output_n, output_n_ro = self.gcn1(data['data_n'], data['adj_n'])
        data_c = torch.bmm(data['h'].transpose(1, 2), output_n)
        output_c, output_c_ro = self.gcn2(data_c, data['adj_c'])
        output = torch.cat((output_n_ro, output_c_ro), dim=1)
        x = self.mlp(output)

        return x

