


import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, VGAE
import streamlit as st
import networkx as nx
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Define GCN, GAT, and GAE classes here

def load_data():
    G = nx.fast_gnp_random_graph(100, 0.05)
    num_nodes = len(G.nodes)
    # Generate node labels
    import random
    node_labels = [random.randint(0, 2) for _ in range(num_nodes)]  # Example node labels
    # Convert NetworkX graph to PyTorch Geometric Data object
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.randn(num_nodes, 16)  # Random node features of size 16
    data = Data(x=x, edge_index=edge_index)
    
    # Set node labels
    y = torch.tensor(node_labels, dtype=torch.long)
    data.y = y
    return data



def single_modality_RL():
    st.title("Node Classification with Graph Neural Networks")
    # Load data
    data = load_data()
    
    # Train/test split
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[:int(data.num_nodes * 0.8)] = 1  # Use 80% of nodes for training
    test_mask = ~train_mask
    # Select method
    method = st.sidebar.selectbox("Select Method", ["GCN", "GAT", "GAE"])
    if method == "GCN":
        model = GCN(input_dim=data.num_features, hidden_dim=16, output_dim=5)
    elif method == "GAT":
        model = GAT(input_dim=data.num_features, hidden_dim=16, output_dim=5)
    elif method == "GAE":
        model = GAE(input_dim=data.num_features, hidden_dim=16, output_dim=5)
    
    # Step 3: Train Model (assuming labels are already present)
    # Here, we'll simply train the model on the Karate Club dataset without any split.
    # In practice, you should split your data into training and validation sets.
    
    model = GCN(input_dim=data.num_features, hidden_dim=16, output_dim=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Training
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])  # Use only training nodes for loss computation
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(data)
        pred = logits.argmax(dim=1)
        test_correct = pred[test_mask] == data.y[test_mask]
        test_acc = int(test_correct.sum()) / int(test_mask.sum())
    st.write("Test Accuracy: {:.4f}".format(test_acc)) 
    
    
    # Compute and plot the confusion matrix
    cm = confusion_matrix(data.y[test_mask], pred[test_mask])
    
    # Plot confusion matrix with percentages
    plt.figure(figsize=(8, 6))
    cm_normalized = cm.astype('float') *100 / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(3))
    plt.yticks(np.arange(3))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, "{:0.2f}".format(cm_normalized[i, j]),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > cm_normalized.max() / 2. else "black")
    plt.tight_layout()
    st.pyplot(plt)
    
    
if __name__ == "__main__":
    main()



