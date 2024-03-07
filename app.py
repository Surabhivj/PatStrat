from functions import GCN, GAT, GAE
import torch.nn.functional as F

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time  # Import the time module for sleeping
import networkx as nx
from sklearn.feature_selection import mutual_info_regression
import numpy as np

import matplotlib.pyplot as plt
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
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    st.set_page_config(layout="wide")  # Set the layout to wide for full-screen display
    st.title("PatStrat: A Framework for Embedding Multi-Modal Patient Similarity Network for Tailor-made  Stratification")
    # Initialize a list to store uploaded files
    uploaded_files = []
    Inputs = []
    processed_dfs = []
    Glist = []
    # Handle tab selection
    Inputs = view_uploaded_files_tab(uploaded_files)
    st.write('### Exploration of individual Modalities: Unsupervised Approach:')
    
    if len(Inputs) == len(uploaded_files):
        processed_dfs = view_preProcessing_dfs(Inputs)
    
    st.write('### Patient Similarity Network (Per Modality):')
    
    if len(processed_dfs) == len(Inputs):
        Glist = view_inferring_networks(processed_dfs)
    
    st.write('### Drug Response Prediction (Per Modality): Supervised Approach:')
    
    if Glist is not None:
        if len(Glist) == len(processed_dfs) - 1:
            view_single_modality_RL(Glist,processed_dfs)
    
    st.write('### Drug Response Prediction (Multi-Modal): Supervised Approach:')
    
    
    #view_inferring_networks(processed_dfs)
    
    
######################################   first row in app   ######################################################3


def view_uploaded_files_tab(uploaded_files):
    #st.header("View Uploaded Files Tab")
    st.sidebar.image("CSEM_logo.png", use_column_width=True)
    # Ask user for the number of files to upload
    num_files = st.sidebar.number_input("Number of Files to Upload (Files per modality)", min_value=1, max_value=10, value=1)
    # Create file uploaders based on the user's input
    for i in range(num_files):
        uploaded_file = st.sidebar.file_uploader(f"Modality {i+1}", type=["csv"])
        if uploaded_file is not None:
            # Append uploaded file to the list
            uploaded_files.append(uploaded_file)
    
    ground_truth = st.sidebar.file_uploader(f"Ground Truth", type=["csv"])
    uploaded_files.append(ground_truth)
    # Display all uploaded files in one row
    st.write("## Biological Question: Prediction of Drug Responses with Multi-Modal Biological Data Integration?")
    st.write("### Modalities")
    columns = st.columns(int(num_files + 1))
    Input_dfs = []
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file is not None:
            # Display file details
            with columns[i % int(num_files + 1)]:
                # Display file content
                if uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file, index_col=0)
                    st.write(uploaded_file.name)
                    st.write(df)
                    Input_dfs.append(df)
                else:
                    st.write("File type not supported. Please upload a CSV file.")
    Input_dats = []
    for df in Input_dfs:
        ground_truth_dat = Input_dfs[-1] 
        subset_df = df[df.columns.intersection(ground_truth_dat.columns)]
        if not subset_df.empty:
            #st.write(uploaded_file.name)
            #st.write(subset_df)
            Input_dats.append(subset_df)
        else:
            st.warning(f"Ignoring DataFrame from {uploaded_file.name} because it does not contain any columns present in the ground truth DataFrame.")
            
    return Input_dats


######################################   second row  in app  ######################################################


def view_preProcessing_dfs(Input_dfs):
    processed_dfs = []
    
    if Input_dfs is not None:
        num_columns = len(Input_dfs)
        if num_columns > 0:
            columns = st.columns(num_columns)
            #st.write(num_columns)
            # Process each DataFrame
            for idx, df in enumerate(Input_dfs):
                with columns[idx]:
                    if idx < (len(Input_dfs) - 1):
                        processed_df = process_single_df(df)
                        processed_dfs.append(processed_df)
                        # Dropdown menu to select manifold learning method
                        ground_truth_dat = Input_dfs[-1]
                        if not ground_truth_dat.empty:
                            ground_truth_processedDat = ground_truth_dat.apply(normalize_row, axis=1)
                            ground_truth_processedDat = ground_truth_processedDat.applymap(factorize_)
                            
                            selction_cols = st.columns(2)
                            method_name = selction_cols[0].selectbox(f'Select Manifold Learning Method', ['Isomap', 'LocallyLinearEmbedding', 'TSNE', 'MDS', 'SpectralEmbedding'], key=f"method_{idx}")
                            selected_drug = selction_cols[1].selectbox('Select Drug', tuple(ground_truth_processedDat.index), key=f"sel_drug_{idx}")
                            
                            colors = {'High': '#AB238C','Moderate': '#27B1A3','Low': '#D2A435', None: 'grey'}
                            ground_truth_processedDat.replace(colors, inplace=True)
                            
                            color_map = dict(ground_truth_processedDat.loc[selected_drug])
                            
                            # Slider bars to adjust parameters
                            data = processed_df.dropna().transpose()
                            
                            if method_name == 'Isomap' or method_name == 'LocallyLinearEmbedding':
                                n_neighbors = st.slider('Number of Neighbors', min_value=2, max_value=20, value=10, key=f"n_neighbors_{idx}")
                                
                                embDat = test_manifold_learning(data, method_name, color_map =color_map , n_neighbors=n_neighbors, random_state=123)
                            elif method_name == 'TSNE':
                                perplexity = st.slider('Perplexity', min_value=5, max_value=50, value=2, key=f"perplexity_{idx}")
                                
                                embDat = test_manifold_learning(data, method_name, color_map=color_map, perplexity=perplexity,random_state=234)
                            elif method_name == 'MDS':
                                
                                embDat = test_manifold_learning(data, method_name, color_map=color_map, random_state=345)
                            elif method_name == 'SpectralEmbedding':
                                
                                embDat = test_manifold_learning(data, method_name, color_map=color_map, random_state=456)
                        
                        # Add download button
                        csv_data = embDat.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Data as CSV",
                            data=csv_data,
                            file_name=f'{method_name}_{selected_drug}_data.csv',
                            mime='text/csv', key=f"emb_{idx}")
                        
                    elif idx == (num_columns - 1):
                        processed_df = df.apply(normalize_row, axis=1)
                        processed_df = processed_df.applymap(factorize_)
                        selected_DR = st.selectbox('Select Drug', tuple(processed_df.index))
                        fig = plot_drug_response(processed_df, selected_DR)
                        st.pyplot(fig)
                        processed_dfs.append(processed_df)
                    else:
                        st.warning(f"Check input data")
    return processed_dfs


######################################   third row  in app ######################################################


def view_inferring_networks(processed_dfs):
    
    if processed_dfs is not None:
        num_columns = len(processed_dfs)
        if num_columns > 0:
            columns = st.columns(num_columns)
            Glist = []
            for i, df in enumerate(processed_dfs):
                if df is not None:
                    if i < (num_columns - 1):
                        df = df.apply(normalize_row, axis=1)
                        df = df.corr(method=custom_mi_reg)
                        threshold = columns[i].slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key=f"thresh_{i}")
                        df[abs(df) < threshold] = 0
                        np.fill_diagonal(df.values, 0)
                        G = nx.from_pandas_adjacency(df)
                        Glist.append(G)
                        # Display network
                        with columns[i]:
                            net = Network(height='400px', width='100%', bgcolor='#222222', font_color='white')
                            net.from_nx(G)
                            net.set_options('''var options = {"physics": {"enabled": false}}''')
                            try:
                                path = '/tmp'
                                net.save_graph(f'{path}/pyvis_graph.html')
                                HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
                            except:
                                path = '/html_files'
                                net.save_graph(f'{path}/pyvis_graph.html')
                                HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
                            components.html(HtmlFile.read(), height=435)
                            
                            # Add download button
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            columns[i].download_button(
                                label="Download Adjacency Matrix as CSV",
                                data=csv_data,
                                file_name='Network_adjacency.csv',
                                mime='text/csv', key=f"adj_{i}")
            return Glist

######################################   fourth row  in app ######################################################

def view_single_modality_RL(Glist, processed_dfs):
    if Glist is not None:
        num_columns = len(Glist)
        if num_columns > 0:
            columns = st.columns(num_columns)
            for i, G in enumerate(Glist):
                if G is not None:
                    with columns[i % int(num_columns)]:
                        GroundTruthDat = processed_dfs[-1]
                        num_nodes = len(G.nodes)
                        cols = st.columns(2)
                        method = cols[0].selectbox("Select Method", ["GCN", "GAT", "GAE"], key=f"selc_method_{i}")
                        drug = cols[1].selectbox('Select Drug', tuple(GroundTruthDat.index), key=f"seldrug_{i}")
                        df = GroundTruthDat.transpose()[drug]
                        response_dict = dict({'Low': 1, 'Moderate': 2, 'High': 3})
                        df = df.replace(response_dict).fillna(4)
                        # Add attributes to nodes from DataFrame (if necessary)
                        # Iterate over each node and its cluster label
                        for node, cl in df.to_dict().items():
                            # Check if the node exists in the graph
                            if node in G.nodes:
                                # Node exists, update its cluster attribute
                                G.nodes[node]['class_label'] = cl
                        # Initialize an empty list to store 'class_label' attributes
                        class_labels = []
                        # Iterate over nodes in the graph
                        for node, attributes in G.nodes(data=True):
                            # Check if the 'class_label' attribute exists for the node
                            if 'class_label' in attributes:
                                # If the attribute exists, append its value to the list
                                class_label = attributes['class_label']
                                class_labels.append(class_label)
                        # Print the list of 'class_label' attributes
                        #st.write(class_labels)
                        node_to_index = {node: index for index, node in enumerate(G.nodes)}
                        # Get the edge indices using the integer indices
                        edges = list(G.edges())
                        source_nodes, target_nodes = zip(*edges)
                        source_indices = [node_to_index[node] for node in source_nodes]
                        target_indices = [node_to_index[node] for node in target_nodes]
                        
                        edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
                        num_nodes = len(G.nodes)
                        x = torch.randn(num_nodes, 16)  # Random node features of size 16
                        # Create a PyTorch geometric Data object
                        data = Data(x=x, edge_index=edge_index)
                        
                        y = torch.tensor(class_labels, dtype=torch.long)
                        data.y = y
                        #st.write(data.y)
                        if st.button("Train Model", key = f'train_button{i}'):
                            single_modality_RL(data, method)


################# Functions that need to be moved to another file ##############################################


def single_modality_RL(data, method):
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

    plt.title('Metrics across 10 runs')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    
    

# Define the plot_drug_response function
def plot_drug_response(dat, selected_DR):
    # Assuming you have already imported normalized_df and want to plot the value counts for the specified row and column
    # High =  '#AB238C'
    # moderate = '#27B1A3'
    # low = '#D2A435'
    
    colors = {'High': '#AB238C','Moderate': '#27B1A3','Low': '#D2A435'}
    row_data = dat.loc[selected_DR].value_counts()
    # Creating the bar plot
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(row_data.index, row_data.values, color=[colors[val] for val in row_data.index])
    # Adding labels and title
    ax.set_xlabel('Response Category', fontsize=12)
    ax.set_ylabel('No. of Patients', fontsize=12)
    ax.set_title(f'Response for {selected_DR}', fontsize=14)
    # Adding gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Display percentage on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{(height / row_data.sum() * 100):.2f}%", ha='center', va='bottom', fontsize=10)
    # Display the plot
    ax.tick_params(axis='x', rotation=0, labelsize=12)  # Rotate x-axis labels for better readability
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    # Return the plot
    return fig


def process_single_df(df):
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
    #df = df.corr(method=custom_mi_reg)  # Example processing, replace with your actual processing logic
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
    if (df > 0.7):
        return 'High'
    elif (df >= 0.3):
        return 'Moderate'
    elif (df < 0.3):
        return 'Low'

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











if __name__ == "__main__":
    main()
