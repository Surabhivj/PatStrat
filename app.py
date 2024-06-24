from functions import GCN, GAT
import functions as FUN
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
import matplotlib.lines as mlines
from pyvis.network import Network
import altair as alt
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random


def main():
    st.set_page_config(layout="wide")  # Set the layout to wide for full-screen display
    st.title("PatStrat: Embedding Multi-Modal Patient Similarity Network for Tailor-made  Stratification")
    col_1, col_3, col_5 = st.columns(3)
    with col_3:
        image_path = "pipeline.png"  # Path to your image file
        st.image(image_path, caption="Pipeline Image", use_column_width=True)
    
    # Initialize a list to store uploaded files
    uploaded_files = []
    Inputs = []
    processed_dfs = []
    Glist = []
    
    # Handle tab selection
    num_files = st.number_input("Modalities", min_value=1, max_value=10, value=3)
    cols = st.columns(num_files)
    for col_index, col in enumerate(cols):
        # Generate a unique key for each uploader based on the column index and file index
        key = f'upload_{col_index}'
        uploaded_file = col.file_uploader(f"Modality {col_index+1}", type=["csv"], key=key)
        if uploaded_file is not None:
            # Append uploaded file to the list
            uploaded_files.append(uploaded_file)

    ground_truth = st.file_uploader("Ground Truth", type=["csv"], key='upload_ground_truth')
    if ground_truth is not None:
        ground_truth = ground_truth

    if uploaded_files and ground_truth is not None:
        col1,col2,col3 = st.columns(3)
        with col1:
            st.write('**Exploration of individual Modalities**')
            file_names = [file.name for file in uploaded_files]
            ground_truth_dat = pd.read_csv(ground_truth,index_col=0)
            drugs = ground_truth_dat.index
            selction_cols = st.columns(3)
            modality = selction_cols[0].selectbox(f'Modality', file_names, key=f"modality")
            method_name = selction_cols[1].selectbox(f'Dimension Reduction Method', ['Isomap', 'TSNE', 'MDS', 'SpectralEmbedding'], key=f"method")
            selected_drug = selction_cols[2].selectbox('Select Drug',drugs , key="sel_drug")
            
            if modality and method_name and selected_drug is not None:
                c1,c2,c3,c4 = st.columns(4)
                #c1.write('ON_Target: Red')
                #c2.write('NO_Effect: White')
                #c3.write('OFF_target: Blue')
                #c4.write('None: Grey')
                for file in uploaded_files:
                    if file.name == modality:
                        uploaded_file = file
                if method_name == 'Isomap':
                    val = st.slider('Number of Neighbors', min_value=2, max_value=20, value=10, key=f"n_neighbors")
                else:
                    val = 1
                if method_name == 'TSNE':
                    val = st.slider('Perplexity', min_value=5, max_value=50, value=2, key=f"perplexity")
                else:
                    val = 1
                embDat = view_preProcessing_dfs(uploaded_file = uploaded_file,method_name = method_name,ground_truth = ground_truth_dat,drug = selected_drug, val =val)
                # Add download button
                if embDat is not None:
                    csv_data = embDat.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv_data,
                        file_name=f'{method_name}_{selected_drug}_data.csv',
                        mime='text/csv', key=f"emb")
                    
        with col2:
            if uploaded_files is not None:
                st.write('**Inferring Patient similarity Network**')
                edge_score_method = 'Mutual_Information'
                edge_score_method = st.selectbox(f'Edge similarity method', ['Mutual_Information', 'Correlation'], key=f"edge_score_method")
                threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key=f"thresh")
                Net = view_inferring_networks(uploaded_files=uploaded_files, method=edge_score_method, score_thresh=threshold)
        with col3:
            st.write('**Network Embedding with Graph Neural Networks**')
            if ground_truth_dat is not None:
                if Net is not None:
                    gnn_method = st.selectbox(f'GNN method', ["GCN", "GAT"], key=f"gnn_method")
                    resdat = view_single_modality_RL(Net = Net, method = gnn_method, ground_truth_dat = ground_truth_dat)
                    # Count the number of occurrences of '1' in each column
                    c1,c2,c3 = st.columns(3)
                    #colors = {'ON_Target': '#53c8cf','N0_Effect': '#767f99','OFF_Target': '#e0809e'}
                    #c1.write('ON_Target: Red')
                    #c2.write('NO_Effect: White')
                    #c3.write('OFF_target: Blue')
                    
                    #colors = {'ON_Target': '#53c8cf','N0_Effect': '#767f99','OFF_Target': '#e0809e'}

                    # Define colormap and custom color mapping for legend
                    cmap = ['#53c8cf', '#767f99', '#e0809e', '#767f99']

                    # Create the heatmap
                    heatmap = go.Figure(data=go.Heatmap(
                        z=resdat.values,
                        x=resdat.columns,
                        y=resdat.index,
                        colorscale=[[0, cmap[0]], [1/3, cmap[1]], [2/3, cmap[2]], [1, cmap[3]]],
                        showscale=False  # Remove color legend
                    ))
                    # Update layout
                    heatmap.update_layout(
                        title="Patient-Drug Response Heatmap",
                        xaxis=dict(title='Drug'),
                        yaxis=dict(title='Patient', autorange='reversed'),
                        width=800,  # Adjust width
                        height=800,  # Adjust height
                        coloraxis_showscale=False
                    )
                    # Show plot in Streamlit
                    st.plotly_chart(heatmap)
                    

@st.cache_resource
def view_preProcessing_dfs(uploaded_file,method_name,ground_truth,drug, val):
    df = pd.read_csv(uploaded_file, index_col=0)
    df = FUN.process_single_df(df,method_name)
    df = df[df.columns.intersection(ground_truth.columns)]
    ground_truth_dat = FUN.factorize_(ground_truth)
    colors = {'ON_Target': '#53c8cf', 'N0_Effect': '#767f99', 'OFF_Target': '#e0809e', None: 'grey'}
    color_map = dict(ground_truth_dat.loc[drug].replace(colors))
    # Slider bars to adjust parameters
    data = df.dropna().transpose()
    
    if method_name == 'Isomap' or method_name == 'LocallyLinearEmbedding':
        embDat = FUN.test_manifold_learning(data, method_name, color_map =color_map , n_neighbors=val, random_state=123)
    elif method_name == 'TSNE':
        
        embDat = FUN.test_manifold_learning(data, method_name, color_map=color_map, perplexity=val,random_state=234)
    elif method_name == 'MDS':
        
        embDat = FUN.test_manifold_learning(data, method_name, color_map=color_map, random_state=345)
    elif method_name == 'SpectralEmbedding':
        
        embDat = FUN.test_manifold_learning(data, method_name, color_map=color_map, random_state=456)
    return embDat

######################################   third row  in app ######################################################
#@st.cache_resource
def view_inferring_networks(uploaded_files, method, score_thresh):
    Glist = []
    for file in uploaded_files:
        df = pd.read_csv(file, index_col=0)
        df = FUN.process_single_df(df,method)
        df = df.apply(FUN.normalize_row, axis=1)
        if method == 'Correlation':
            df = df.corr(method=FUN.custom_mi_reg)
        if method == 'Mutual_Information':
            df = df.corr()
        df[abs(df) < score_thresh] = 0
        np.fill_diagonal(df.values, 0)
        G = nx.from_pandas_adjacency(df)
        Glist.append(G)

    Nets = nx.compose_all(Glist)
    print(Nets.nodes())
    if 'Unnamed: 139' in Nets.nodes:
        Nets.remove_node('Unnamed: 139')
    # Display network
    net = Network(height='400px', width='100%', bgcolor='#222222', font_color='white')
    net.from_nx(Nets)
    net.set_options('''var options = {"physics": {"enabled": false}}''')
    try:
        path = '/tmp'
        net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
    except:
        path = '/html_files'
        net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
    yellow_background = "<style>:root {background-color: black;}</style>"
    components.html(yellow_background + HtmlFile.read(), height=410)
    return Nets

######################################   fourth row  in app ######################################################
def view_single_modality_RL(Net, method, ground_truth_dat):
    num_nodes = len(Net.nodes)
    resdat = pd.DataFrame()
    
    ground_truth_dat = FUN.factorize_(ground_truth_dat)
    colors = {'ON_Target': 'red', 'N0_Effect': 'white', 'OFF_Target': 'blue', None: 'grey'}
    ground_truth_dat.replace(colors, inplace = True)
    
    for drug in ground_truth_dat.index:
        df = ground_truth_dat.transpose()[drug]
        common_nodes = set(Net.nodes).intersection(set(df.index))
        # Create a subset of Net containing only the common nodes
        subset_net = Net.subgraph(common_nodes)
        Net = subset_net
        response_dict = dict({'red': 1, 'white': 2, 'blue': 3, 'grey': 4})
        
        df = df.replace(response_dict).fillna(4)
        # Add attributes to nodes from DataFrame (if necessary)
        # Iterate over each node and its cluster label
        for node, cl in df.to_dict().items():
            # Check if the node exists in the graph
            if node in Net.nodes:
                # Node exists, update its cluster attribute
                Net.nodes[node]['class_label'] = cl
        # Initialize an empty list to store 'class_label' attributes
        class_labels = []
        # Iterate over nodes in the graph
        for node, attributes in Net.nodes(data=True):
            # Check if the 'class_label' attribute exists for the node
            if 'class_label' in attributes:
                # If the attribute exists, append its value to the list
                class_label = attributes['class_label']
                class_labels.append(class_label)
        # Print the list of 'class_label' attributes
        #st.write(class_labels)
        node_to_index = {node: index for index, node in enumerate(Net.nodes)}
        # Get the edge indices using the integer indices
        edges = list(Net.edges())
        source_nodes, target_nodes = zip(*edges)
        source_indices = [node_to_index[node] for node in source_nodes]
        target_indices = [node_to_index[node] for node in target_nodes]
        edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
        num_nodes = len(Net.nodes)
        x = torch.randn(num_nodes, 16)  # Random node features of size 16
        # Create a PyTorch geometric Data object
        data = Data(x=x, edge_index=edge_index)
        y = torch.tensor(class_labels, dtype=torch.long)
        data.y = y
        test_mask, ypred = FUN.single_modality_RL(data, method,drug)
        testdat = pd.DataFrame({'nodes':list(Net.nodes), 'test': test_mask})
        nodes = testdat[testdat['test'] == True]['nodes']
        resdat.index = nodes
        resdat[drug] = ypred

        #print(resdat)

    return resdat


if __name__ == "__main__":
    main()
