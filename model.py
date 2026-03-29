import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net(torch.nn.Module):

    def __init__(self, in_channels, hidden=128, dropout=0.3):
        super().__init__()

        # ✅ SINGLE shared GCN (lightweight)
        self.conv = GCNConv(in_channels, hidden)
        self.dropout = dropout
        # ✅ Smaller MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden * 3, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1)
        )
    # -------------------------
    # Encode (Multi-Kernel)
    # -------------------------
    def encode(self, x, edge_index_I, edge_index_D):

        # SAME GCN for both kernels (key idea)
        h_I = self.conv(x, edge_index_I)
        h_I = F.relu(h_I)
        h_I = F.dropout(h_I, p=self.dropout, training=self.training)
        h_D = self.conv(x, edge_index_D)
        h_D = F.relu(h_D)
        h_D = F.dropout(h_D, p=self.dropout, training=self.training)

        # SIMPLE ADD (same as paper)
        h = h_I + h_D

        return h

    # -------------------------
    # Decode
    # -------------------------
    def decode(self, z, edge_label_index):

        zi = z[edge_label_index[0]]
        zj = z[edge_label_index[1]]

        pair = torch.cat(
            [zi, zj, torch.abs(zi - zj)],
            dim=1
        )

        return self.mlp(pair).squeeze()

    # -------------------------
    def forward(self, data):

        z = self.encode(
            data.x,
            data.edge_index,
            data.edge_index_D
        )

        return self.decode(z, data.edge_label_index)
# =====================================
# HEAVY BASELINE MODEL (for comparison)
# =====================================
class HeavyNet(torch.nn.Module):

    def __init__(self, in_channels, hidden=256):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden * 3, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        zi = z[edge_label_index[0]]
        zj = z[edge_label_index[1]]

        pair = torch.cat([zi, zj, torch.abs(zi - zj)], dim=1)
        return self.mlp(pair).squeeze()    