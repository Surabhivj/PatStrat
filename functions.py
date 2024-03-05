import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, VGAE
import streamlit as st
import networkx as nx


# Define GCN, GAT, and GAE classes here
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8)
        self.conv2 = GATConv(8 * hidden_dim, output_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAE, self).__init__()
        self.encoder = GCNConv(input_dim, hidden_dim)
        self.vgae = VGAE(self.encoder, decoder=None)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z, mu, logvar = self.vgae(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z

        