from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GINConv, global_mean_pool
import torch
import torch
import torch.nn as nn
from torch_geometric.data import Data

class GCNNet(torch.nn.Module):
    def __init__(self, num_feature, num_label, hidden_dim=16, dropout=0.6):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_feature, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_label)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def forward_with_weights(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, num_feature, num_label, hidden_dim=16, heads=8, dropout=0.6):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(
            in_channels=num_feature,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout
        )
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=num_label,
            heads=1,
            concat=False,
            dropout=dropout
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def forward_with_weights(self, x, edge_index, edge_weight=None):

        if edge_weight is not None:
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            return self.forward_with_gat_weights(x, edge_index, edge_weight)
        else:
            data = Data(x=x, edge_index=edge_index)
            return self.forward(data)

    def forward_with_gat_weights(self, x, edge_index, edge_weight):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)


        x = self.apply_weighted_gat(x, edge_index, edge_weight, self.conv2)

        return F.log_softmax(x, dim=1)

    def apply_weighted_gat(self, x, edge_index, edge_weight, conv_layer):

        lin_weight = conv_layer.lin_src.weight
        att_src = conv_layer.att_src
        att_dst = conv_layer.att_dst


        x_src = torch.matmul(x, lin_weight)
        alpha_src = (x_src * att_src).sum(dim=-1)
        alpha_dst = (x_src * att_dst).sum(dim=-1)


        row, col = edge_index
        alpha = alpha_src[row] + alpha_dst[col]
        alpha = F.leaky_relu(alpha, conv_layer.negative_slope)


        alpha = alpha * edge_weight


        alpha = softmax(alpha, row, x.size(0))


        alpha = F.dropout(alpha, p=conv_layer.dropout, training=self.training)


        out = torch.zeros_like(x_src)
        out = out.scatter_add_(0, col.unsqueeze(-1).expand(-1, x_src.size(-1)),
                               x_src[row] * alpha.unsqueeze(-1))

        if conv_layer.bias is not None:
            out += conv_layer.bias

        return out


def softmax(src, index, num_nodes=None):

    if num_nodes is None:
        num_nodes = index.max().item() + 1


    max_value = src.max()
    exp_src = torch.exp(src - max_value)


    sum_exp = torch.zeros(num_nodes, device=src.device)
    sum_exp = sum_exp.scatter_add_(0, index, exp_src)


    return exp_src / sum_exp[index]


class GINNet(torch.nn.Module):
    def __init__(self, num_feature, num_label, hidden_dim=32, dropout=0.6):
        super(GINNet, self).__init__()

        nn1 = nn.Sequential(
            nn.Linear(num_feature, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gin1 = GINConv(nn1)

        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_label)
        )
        self.gin2 = GINConv(nn2)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gin1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gin2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def forward_with_weights(self, x, edge_index, edge_weight=None):

        if edge_weight is not None:

            x = F.relu(self.gin1(x, edge_index, edge_weight=edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.gin2(x, edge_index, edge_weight=edge_weight)
            return F.log_softmax(x, dim=1)
        else:
            data = Data(x=x, edge_index=edge_index)
            return self.forward(data)
