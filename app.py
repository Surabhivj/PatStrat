import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time  # Import the time module for sleeping
import networkx as nx
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from pyvis.network import Network
import matplotlib.pyplot as plt
from sklearn import manifold, cluster
import mplcursors
import matplotlib.lines as mlines

def main():
    st.set_page_config(layout="wide")  # Set the layout to wide for full-screen display
    st.title("PatStrat: A Framework for Embedding Multi-Modal Patient Similarity Network for Tailor-made  Stratification")
    # Initialize a list to store uploaded files
    uploaded_files = []
    Inputs = []
    processed_dfs = []
    # Handle tab selection
    Inputs = view_uploaded_files_tab(uploaded_files)
    
    if len(Inputs) == len(uploaded_files):
        processed_dfs = view_preProcessing_dfs(Inputs)
    #view_inferring_networks(processed_dfs)
    
    
######################################   first row in app   ######################################################3


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

######################################   second row  in app ######################################################


def view_preProcessing_dfs(Input_dfs):
    processed_dfs = []
    
    if Input_dfs is not None:
        num_columns = len(Input_dfs)
        if num_columns > 0:
            columns = st.columns(num_columns)
            st.write(num_columns)
            
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
                                
                                test_manifold_learning(data, method_name, color_map =color_map , n_neighbors=n_neighbors, random_state=123)
                            elif method_name == 'TSNE':
                                perplexity = st.slider('Perplexity', min_value=5, max_value=50, value=2, key=f"perplexity_{idx}")
                                
                                test_manifold_learning(data, method_name, color_map=color_map, perplexity=perplexity,random_state=234)
                            elif method_name == 'MDS':
                                
                                test_manifold_learning(data, method_name, color_map=color_map, random_state=345)
                            elif method_name == 'SpectralEmbedding':
                                
                                test_manifold_learning(data, method_name, color_map=color_map, random_state=456)
                    
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
    
################# Functions that need to be moved to another file ##############################################

# Define the plot_drug_response function
def plot_drug_response(dat, selected_DR):
    # Assuming you have already imported normalized_df and want to plot the value counts for the specified row and column
    
    # High =  '#AB238C'
    # moderate = '#27B1A3'
    # low = '#D2A435'
    colors = {'High': '#AB238C','Moderate': '#27B1A3','Low': '#D2A435'}
    row_data = dat.loc[selected_DR].value_counts()

    # Get a colormap
    #colors = plt.cm.Set2(np.linspace(0, 1, len(row_data)))

    # Creating the bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
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
    

def test_manifold_learning(data, method_name, color_map, n_neighbors=10, perplexity=2, eps=0.2,random_state=123456):
    """
    Test a manifold learning method and plot the result.
    
    Parameters:
        data (array-like): Input data matrix of shape (n_samples, n_features).
        method_name (str): Name of the manifold learning method.
        n_neighbors (int): Number of neighbors parameter (used in some methods).
        perplexity (int): Perplexity parameter (used in TSNE).
        eps (float): Maximum distance between two samples for one to be considered as in the neighborhood of the other (DBSCAN parameter).
        min_samples (int): Number of samples in a neighborhood for a point to be considered as a core point (DBSCAN parameter).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if method_name == 'Isomap':
        embedding = manifold.Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(data)
    elif method_name == 'LocallyLinearEmbedding':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, random_state=random_state, method='standard').fit_transform(data)
    elif method_name == 'TSNE':
        embedding = manifold.TSNE(n_components=2, init='pca', perplexity=perplexity,random_state=random_state).fit_transform(data)
    elif method_name == 'MDS':
        embedding = manifold.MDS(n_components=2, random_state=random_state).fit_transform(data)
    elif method_name == 'SpectralEmbedding':
        embedding = manifold.SpectralEmbedding(n_components=2, random_state=random_state).fit_transform(data)
        

    colors_reversedict = {'#AB238C' : 'High Response', '#27B1A3': 'Moderate Response', '#D2A435' : 'Low Response', 'grey': 'NA'}

    # Perform k-means clustering
    kmeans = cluster.KMeans(n_clusters=3, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embedding)

    # Define markers for each cluster
    markers = ['o', 's', '^']

    # Plot the embedding with colors and different markers for each cluster
    for i, marker in enumerate(markers):
        cluster_mask = (cluster_labels == i)
        ax.scatter(embedding[cluster_mask, 0], embedding[cluster_mask, 1], c=[color_map[ID] for ID in data.index[cluster_mask]], marker=marker, label=f'Cluster {i + 1}')

    # Add legend for markers and colors
    unique_colors = list(set([color_map[ID] for ID in data.index]))
    marker_legend_handles = [mlines.Line2D([], [], color='black', marker=marker, linestyle='None', markersize=10, label=f'Cluster {i + 1}') for i, marker in enumerate(markers)]
    color_legend_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=colors_reversedict[color]) for color in unique_colors]
    handles = marker_legend_handles + color_legend_handles
    ax.legend(handles=handles, title='Legend', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(handles))


    plt.title(f'{method_name}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    
    # Enable hover functionality
    mplcursors.cursor(hover=True)
    
    # Display the plot in Streamlit
    st.pyplot(plt.gcf())



if __name__ == "__main__":
    main()
