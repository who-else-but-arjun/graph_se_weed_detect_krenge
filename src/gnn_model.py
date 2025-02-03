import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_classes=2):
        super(GNN, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=8, dropout=0.5)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=8, dropout=0.5)
        self.conv3 = GCNConv(hidden_dim * 8, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim * 8)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim * 8)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        self.training_step = 0
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.elu(x)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.elu(x)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.elu(x)
        
        x = self.lin(x)
        return F.log_softmax(x, dim=1)
