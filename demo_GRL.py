from functions import GCN, GAT, GAE
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


st.set_page_config(layout="wide") 
st.title('PatStrat: Embedding Multi-Modal Patient Similarity Networks for Drug Response Stratification')

col_1, col_3, col_5 = st.columns(3)
with col_3:
    image_path = "pipeline.png"  # Path to your image file
    st.image(image_path, caption="Pipeline Image", use_column_width=True)

############################################ FILE UPLOAD ##############################################################
uploaded_files = []

# Create 5 columns
col1, col2, col3, col4, col5 = st.columns(5)

filenames = ["Proteomics-CD138_Cells", "Proteomics-Myeloma_Cells", "Clinical",  "Cytokines", "Drug Responses"]

# Within each column, place a file uploader
with col1:
    uploaded_proteomics_cd138_cells = st.file_uploader("Proteomics-CD138_Cells", type='csv', key='file_uploader_1', accept_multiple_files=False)
    if uploaded_proteomics_cd138_cells is not None:
        uploaded_files.append(uploaded_proteomics_cd138_cells)

with col2:
    uploaded_proteomics_myeloma_cells = st.file_uploader("Proteomics-Myeloma_Cells", type='csv', key='file_uploader_2', accept_multiple_files=False)
    if uploaded_proteomics_myeloma_cells is not None:
        uploaded_files.append(uploaded_proteomics_myeloma_cells)

with col3:
    uploaded_clinical = st.file_uploader("Clinical", type='csv', key='file_uploader_3', accept_multiple_files=False)
    if uploaded_clinical is not None:
        uploaded_files.append(uploaded_clinical)

with col4:
    uploaded_cytokines = st.file_uploader("Cytokines", type='csv', key='file_uploader_4', accept_multiple_files=False)
    if uploaded_cytokines is not None:
        uploaded_files.append(uploaded_cytokines)

with col5:
    uploaded_drug_responses = st.file_uploader("Drug Responses", type='csv', key='file_uploader_5', accept_multiple_files=False)
    if uploaded_drug_responses is not None:
        uploaded_files.append(uploaded_drug_responses)


############################################ DISPLAY FILE UPLOAD ##############################################################

if len(uploaded_files) == 5:
    # Create 5 columns
    cols = st.columns(5)
    
    for idx, file in enumerate(uploaded_files):
        cols[idx].write(filenames[idx])
        df = pd.read_csv(file)
        cols[idx].write(df)
        
######################################### Infer Multilayer Graph #########################################################
        
    patients = df.columns
    edge_score = st.selectbox(f'Select similarity measure to infer Multilayer network', ['Correlation', 'Mutual_Information'], key=f"edge_score_method")
    
    def infer_multiNet(uploaded_files, method):
        if method == Correlation:
            for idx, file in enumerate(uploaded_files):
                df = pd.read_csv(file)
                cor_mat = df.corr()
                
    ###################################### Manifold #####################################
    
    if edge_score:

        G = FUN.net_vis()
        
        # Create a Pyvis network
        net = Network(height='500px', width='500px', bgcolor='#252729', font_color='white')
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

