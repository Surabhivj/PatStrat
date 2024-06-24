import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, VGAE
import streamlit as st
import networkx as nx
import torch.nn.functional as F
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time  # Import the time module for sleeping
import networkx as nx
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from sklearn import manifold, cluster
import mplcursors
import matplotlib.lines as mlines
from pyvis.network import Network
import altair as alt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, VGAE
import streamlit as st
import networkx as nx
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dgl.nn.pytorch import SAGEConv


# class GraphSAGE(nn.Module):
#     def __init__(self, in_feats, hidden_feats, out_feats, num_layers, activation):
#         super(GraphSAGE, self).__init__()
#         self.layers = nn.ModuleList()
#         self.activation = activation
#         # Input layer
#         self.layers.append(SAGEConv(in_feats, hidden_feats, 'mean'))
#         # Hidden layers
#         for _ in range(num_layers - 2):
#             self.layers.append(SAGEConv(hidden_feats, hidden_feats, 'mean'))
#         # Output layer
#         self.layers.append(SAGEConv(hidden_feats, out_feats, 'mean'))

#     def forward(self, g, inputs):
#         h = inputs
#         for layer in self.layers[:-1]:
#             h = layer(g, h)
#             h = self.activation(h)
#         h = self.layers[-1](g, h)
#         return h


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


# class GAE(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GAE, self).__init__()
#         self.encoder = GCNConv(input_dim, hidden_dim)
#         self.vgae = VGAE(self.encoder, decoder=None)

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5*logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         z, mu, logvar = self.vgae(x, edge_index)
#         z = self.reparameterize(mu, logvar)
#         return z

def single_modality_RL(data, method,drug):
    # Train/test split
    # Train/test split
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[:int(data.num_nodes * 0.8)] = 1  # Use 80% of nodes for training
    test_mask = ~train_mask
    
    #st.write(len(train_mask))

    if method == "GCN":
        np.random.seed(12)
        model = GCN(input_dim=data.num_features, hidden_dim=16, output_dim=5)
    elif method == "GAT":
        np.random.seed(23)
        model = GAT(input_dim=data.num_features, hidden_dim=16, output_dim=5)
    elif method == "GAE":
        np.random.seed(34)
        model = GAE(input_dim=data.num_features, hidden_dim=16, output_dim=5)
    
    # Train Model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Lists to store metrics for each run
    accuracy_list = []
    precision_list_micro = []
    recall_list_micro = []
    f1_score_list_micro = []
    precision_list_macro = []
    recall_list_macro = []
    f1_score_list_macro = []

    # Perform 10 runs
    for run in range(10):   
        # Training
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out[train_mask], data.y[train_mask])  # Use only training nodes for loss computation
            loss.backward()
            optimizer.step()
        # Evaluation
        model.eval()
        with torch.no_grad():
            logits = model(data)
            pred = logits.argmax(dim=1)

        y_pred =  pred[test_mask]
        y_true = data.y[test_mask]

        # Calculate metrics for the current run
        accuracy = accuracy_score(y_true, y_pred)
        precision_micro = precision_score(y_true, y_pred, average='micro')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        # Store the metrics for the current run
        accuracy_list.append(accuracy)
        precision_list_micro.append(precision_micro)
        recall_list_micro.append(recall_micro)
        f1_score_list_micro.append(f1_micro)
        precision_list_macro.append(precision_macro)
        recall_list_macro.append(recall_macro)
        f1_score_list_macro.append(f1_macro)

    # Calculate mean and standard deviation of each metric
    accuracy_mean = np.mean(accuracy_list)
    precision_mean_micro = np.mean(precision_list_micro)
    recall_mean_micro = np.mean(recall_list_micro)
    f1_score_mean_micro = np.mean(f1_score_list_micro)
    precision_mean_macro = np.mean(precision_list_macro)
    recall_mean_macro = np.mean(recall_list_macro)
    f1_score_mean_macro = np.mean(f1_score_list_macro)

    accuracy_std = np.std(accuracy_list)
    precision_std_micro = np.std(precision_list_micro)
    recall_std_micro = np.std(recall_list_micro)
    f1_score_std_micro = np.std(f1_score_list_micro)
    precision_std_macro = np.std(precision_list_macro)
    recall_std_macro = np.std(recall_list_macro)
    f1_score_std_macro = np.std(f1_score_list_macro)

    # Plot the metrics
    plt.figure(figsize=(10, 6))

    plt.errorbar(['Accuracy', 'Precision (Micro)', 'Recall (Micro)', 'F1 Score (Micro)',
                'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)'],
                [accuracy_mean, precision_mean_micro, recall_mean_micro, f1_score_mean_micro,
                precision_mean_macro, recall_mean_macro, f1_score_mean_macro],
                yerr=[accuracy_std, precision_std_micro, recall_std_micro, f1_score_std_micro,
                    precision_std_macro, recall_std_macro, f1_score_std_macro],
                fmt='o')

    # Add annotations for mean values
    for i, value in enumerate([accuracy_mean, precision_mean_micro, recall_mean_micro, f1_score_mean_micro,
                precision_mean_macro, recall_mean_macro, f1_score_mean_macro]):
        plt.annotate(f'{round(value, 2)}', (i, value))

    plt.title(f'Metrics across 10 runs: {drug}')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.savefig(f"results/Response_{drug}.png")

    return test_mask, y_pred


def plot_drug_response(dat, selected_DR):
    # Define colors for different response categories
    colors = {'ON_Target': '#53c8cf','N0_Effect': '#767f99','OFF_Target': '#e0809e'}
    
    # Get value counts for the selected drug response
    row_data = dat.loc[selected_DR].value_counts().reset_index()
    row_data.columns = ['Response', 'Count']
    
    # Create Altair bar chart
    bars = alt.Chart(row_data).mark_bar().encode(
        x=alt.X('Response:O', title='Response Category'),
        y=alt.Y('Count:Q', title='No. of Patients'),
        color=alt.Color('Response:N', scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values()))),
        tooltip=['Count']  # Show count in tooltip
    ).properties(
        width=300,
        height=300,
        title=f'Response for {selected_DR}'
    )
    
    # Add text on top of bars showing percentage
    text = bars.mark_text(
        align='center',
        baseline='bottom',
        dy=-5  # Nudge text upward slightly to fit within the bar
    ).encode(
        text=alt.Text('Count:Q', format='.2f')
    )
    
    # Combine bars and text
    chart = (bars + text)
    
    # Adjust axis labels and ticks
    chart = chart.properties(
        title=alt.TitleParams(text=f'Response for {selected_DR}', fontSize=14),
        encoding={
            'x': alt.X('Response:O', title='Response Category', axis=alt.Axis(labelFontSize=12, labelAngle=0)),
            'y': alt.Y('Count:Q', title='No. of Patients', axis=alt.Axis(labelFontSize=10))
        }
    )
    
    return chart

# Example usage:
# Assuming you have already imported normalized_df and want to plot the value
def process_single_df(df,method):
    for column in df.columns:
        dtype = df[column].dtype
        if dtype == "object":
            label_encoder = LabelEncoder()
            if type(df[column][0]) == str:
                df[column] = label_encoder.fit_transform(df[column])
        elif dtype in ["int64", "float64"]:
            pass
        else:
            raise ValueError("TypeError")
    df.replace(to_replace='', value=np.nan, inplace=True)
    df.replace(to_replace='None', value=np.nan, inplace=True)
    df.replace(to_replace='NaN', value=np.nan, inplace=True)
    df.replace(to_replace='na', value=np.nan, inplace=True)
    if method == 'Correlation':
        df = df.corr(method=custom_mi_reg)
    if method == 'Mutual_Information':
        df = df.corr()
    return df

def custom_mi_reg(a, b):
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    return  mutual_info_regression(a, b)[0] # should return a float value

# Define custom normalization function
def normalize_row(row):
    min_val = row.min(skipna=True)
    max_val = row.max(skipna=True)
    if min_val == max_val:  # If all values in the row are the same
        return row
    return (row - min_val) / (max_val - min_val)

def factorize_(df):
    def factorize_single_value(val):
        if val > 0.1:
            return 'ON_Target'
        elif -0.1 <= val <= 0.1:
            return 'N0_Effect'
        elif val < -0.1:
            return 'OFF_Target'
    return df.applymap(factorize_single_value)



def test_manifold_learning(dat, method_name, color_map, n_neighbors=10, perplexity=2, random_state=123456):
    """
    Test a manifold learning method and plot the result.

    Parameters:
        data (array-like): Input data matrix of shape (n_samples, n_features).
        method_name (str): Name of the manifold learning method.
        n_neighbors (int): Number of neighbors parameter (used in some methods).
        perplexity (int): Perplexity parameter (used in TSNE).
        random_state (int): Random state.
    """
    # Perform manifold learning
    try:
        if method_name == 'Isomap':
            embedding = manifold.Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(dat)
        elif method_name == 'LocallyLinearEmbedding':
            embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, random_state=random_state, method='standard').fit_transform(dat)
        elif method_name == 'TSNE':
            embedding = manifold.TSNE(n_components=2, init='pca', perplexity=perplexity, random_state=random_state).fit_transform(dat)
        elif method_name == 'MDS':
            embedding = manifold.MDS(n_components=2, random_state=random_state).fit_transform(dat)
        elif method_name == 'SpectralEmbedding':
            embedding = manifold.SpectralEmbedding(n_components=2, random_state=random_state).fit_transform(dat)
        else:
            raise ValueError("Unsupported method_name. Supported values are 'Isomap', 'LocallyLinearEmbedding', 'TSNE', 'MDS', and 'SpectralEmbedding'.")
    except ValueError as e:
        st.error(str(e))
        return

    # Perform k-means clustering
    kmeans = cluster.KMeans(n_clusters=3, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embedding)
    
    # Create a sample dataframe
    data = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'ID': dat.index,
        'KMeans Cluster': cluster_labels + 1
    })
    
    # Map IDs to colors
    data['color'] = data['ID'].map(color_map)

    # Create scatter plot using Altair
    scatter_plot = alt.Chart(data).mark_circle(size = 100).encode(
        x='x',
        y='y',
        color=alt.Color('color', scale=None, legend=alt.Legend(title='Response Level', values=list(color_map.values()), labelOverlap='parity', symbolLimit=200)),
        tooltip=['ID','KMeans Cluster']
        ).properties(
        width=500,
        height=400).configure_legend(orient='bottom').interactive()
    # Render the plot
    st.altair_chart(scatter_plot, use_container_width=True)
    
    return data



def net_vis():
    # Sample data representing a multilayer network
    layers_data = {
        'Layer 1': {
            'nodes': ['A', 'B', 'C'],
            'edges': [('A', 'B'), ('B', 'C'), ('C', 'A')]
        },
        'Layer 2': {
            'nodes': ['A', 'Y', 'Z'],
            'edges': [('A', 'Y'), ('Y', 'Z'), ('Z', 'A')]
        }
        # Add more layers here if needed
    }

    # Create a new NetworkX graph
    G = nx.Graph()

    # Add nodes and edges for each layer
    for layer, data in layers_data.items():
        for node in data['nodes']:
            G.add_node(f"{node}_{layer.replace(' ', '_')}", layer=layer)
        for edge in data['edges']:
            G.add_edge(f"{edge[0]}_{layer.replace(' ', '_')}", f"{edge[1]}_{layer.replace(' ', '_')}")

    # Add visible edges to connect nodes across layers
    for layer1, data1 in layers_data.items():
        for node1 in data1['nodes']:
            for layer2, data2 in layers_data.items():
                if layer1 != layer2:
                    for node2 in data2['nodes']:
                        if node1 == node2:
                            G.add_edge(f"{node1}_{layer1.replace(' ', '_')}", f"{node2}_{layer2.replace(' ', '_')}", color='grey', width=1)
    return G