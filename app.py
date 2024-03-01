import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time  # Import the time module for sleeping
import networkx as nx
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from pyvis.network import Network


def main():
    st.set_page_config(layout="wide")  # Set the layout to wide for full-screen display
    st.title("TITLE")
    # Initialize a list to store uploaded files
    uploaded_files = []
    Input_dfs = []
    processed_dfs = []
    # Handle tab selection
    Input_dfs = view_uploaded_files_tab(uploaded_files)
    processed_dfs = view_preProcessing_dfs(Input_dfs)
    view_inferring_networks(processed_dfs)

def view_uploaded_files_tab(uploaded_files):
    #st.header("View Uploaded Files Tab")
    # Ask user for the number of files to upload
    num_files = st.sidebar.number_input("Number of Files to Upload (Files per modality)", min_value=1, max_value=10, value=1)
    # Create file uploaders based on the user's input
    for i in range(num_files):
        uploaded_file = st.sidebar.file_uploader(f"Modality {i+1}", type=["csv"])
        if uploaded_file is not None:
            # Append uploaded file to the list
            uploaded_files.append(uploaded_file)
    # Display all uploaded files in one row
    st.write("## Modalities")
    columns = st.columns(int(num_files))
    Input_dfs = []
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file is not None:
            # Display file details
            with columns[i % int(num_files)]:
                # Display file content
                if uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file, index_col=0)
                    st.write(uploaded_file.name)
                    st.write(df)
                    Input_dfs.append(df)
                else:
                    st.write("File type not supported. Please upload a CSV file.")    
    return Input_dfs

def view_preProcessing_dfs(Input_dfs):
    processed_dfs = []
    # Add a button to trigger DataFrame processing
    if st.button("Process All DataFrames", key="process_button"):
        with st.spinner("Processing DataFrames..."):
            # Process each DataFrame
            for df in Input_dfs:
                    processed_df = process_single_df(df)
                    processed_dfs.append(processed_df)
                    time.sleep(1)  # Simulate processing time
        st.success("DataFrames Processing Complete!")
    return processed_dfs

def view_inferring_networks(processed_dfs):
    columns = st.columns(int(3))
    for i, df in enumerate(processed_dfs):
        if df is not None:
            df[abs(df) < 0.7] = 0
            np.fill_diagonal(df.values, 0)
            G = nx.from_pandas_adjacency(df)
            
            with columns[i % int(3)]:
                st.write(df)
                net = Network(height='400px',width='100%',bgcolor='#222222',font_color='white')
                net.from_nx(G)
                net.set_options('''var options = {"physics": {"enabled": false}}''')
                try:
                    path = '/tmp'
                    net.save_graph(f'{path}/pyvis_graph.html')
                    HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
                # Save and read graph as HTML file (locally)
                except:
                    path = '/html_files'
                    net.save_graph(f'{path}/pyvis_graph.html')
                    HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
                components.html(HtmlFile.read(), height=435)


def view_integration_nets(nets):
    embeddings = []
    # Add a button to trigger DataFrame processing
    if st.button("Integrate multi-modal data", key="process_integration"):
        with st.spinner("Integrating Modalities using MultiLayer Graph Representation Learning..."):
            # Process each DataFrame
            for net in nets:
                    #processed_df = process_single_df(df)
                    #processed_dfs.append(processed_df)
                    time.sleep(1)  # Simulate processing time
        st.success("Integrating Modalities Complete!")
    #return processed_dfs

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
    df = df.corr(method=custom_mi_reg)  # Example processing, replace with your actual processing logic
    return df

def custom_mi_reg(a, b):
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    return  mutual_info_regression(a, b)[0] # should return a float value

if __name__ == "__main__":
    main()
