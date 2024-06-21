# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import networkx as nx
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree

# Step 1: Data Loading and Preprocessing (Example)
@st.cache  # Cache data loading for better performance
def load_data():
    data = {
        'patient_id': range(100),
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'treatment_line': np.random.choice(['line1', 'line2'], 100),
        'drug_response': np.random.rand(100) * 100  # Example continuous drug response

    }
    df = pd.DataFrame(data)
    return df

# Step 2: Preprocessing Functions
def preprocess_data(df):
    scaler = StandardScaler()
    df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])
    return df, scaler

# Step 3: Graph Operations and Node2Vec Embeddings
def add_new_patient_to_graph(G, new_patient_data, existing_patient_data):
    new_patient_id = new_patient_data['patient_id']
    G.add_node(new_patient_id)
    new_patient_features = new_patient_data[['feature1', 'feature2']].values
    for existing_patient_id in existing_patient_data.index:
        existing_features = existing_patient_data.loc[existing_patient_id, ['feature1', 'feature2']].values
        similarity = np.exp(-np.linalg.norm(new_patient_features - existing_features))
        G.add_edge(new_patient_id, existing_patient_id, weight=similarity)
    return G

def get_node2vec_embeddings(G):
    # Implement Node2Vec embedding generation here
    # Example placeholder for node2vec
    node_ids = list(G.nodes)
    embeddings = {str(i): np.random.rand(64) for i in node_ids}  # Replace with actual embeddings
    return embeddings

# Step 4: Machine Learning Model Training and Prediction
def train_model(df):
    models = {}
    for treatment_line in df['treatment_line'].unique():
        sub_df = df[df['treatment_line'] == treatment_line]
        X = sub_df[['feature1', 'feature2']].values
        y = sub_df['drug_response'].values

        model = LinearRegression()
        model.fit(X, y)
        models[treatment_line] = model
    
    return models

# Step 5: Predict Drug Response for New Patient
def predict_drug_response(new_patient_data, scaler, existing_patient_data, networks, trained_models):
    new_patient_df = pd.DataFrame([new_patient_data])
    new_patient_df[['feature1', 'feature2']] = scaler.transform(new_patient_df[['feature1', 'feature2']])

    G = networks[new_patient_data['treatment_line']]
    G = add_new_patient_to_graph(G, new_patient_df.iloc[0], existing_patient_data)

    new_embeddings = get_node2vec_embeddings(G)
    new_patient_embedding = new_embeddings[str(new_patient_data['patient_id'])]

    # Ensure new_patient_embedding is reshaped to match the model input shape
    new_patient_embedding = new_patient_embedding.reshape(1, -1)  # Assuming 64-dimensional embeddings

    model = trained_models[new_patient_data['treatment_line']]
    predicted_response = model.predict(new_patient_embedding.reshape(1, -1))

    return predicted_response


# Main Streamlit Application
def main():
    st.title('Drug Response Prediction App')

    # Step 1: Data Loading and Preprocessing
    df = load_data()
    df, scaler = preprocess_data(df)

    # Display the loaded dataset
    st.subheader('Example Dataset')
    st.dataframe(df)

    # Step 4: Machine Learning Model Training
    trained_models = train_model(df)

    # Step 3: Graph Operations and Node2Vec Embeddings
    # Example placeholder for networks
    networks = {
        'line1': nx.Graph(),
        'line2': nx.Graph()
    }

    # Step 5: Prediction Form
    st.subheader('Predict Drug Response for New Patient')

    new_patient_id = st.number_input('Enter Patient ID:')
    new_patient_feature1 = st.number_input('Enter Feature 1:')
    new_patient_feature2 = st.number_input('Enter Feature 2:')
    treatment_line = st.selectbox('Select Treatment Line:', df['treatment_line'].unique())

    new_patient_data = {
        'patient_id': new_patient_id,
        'feature1': new_patient_feature1,
        'feature2': new_patient_feature2,
        'treatment_line': treatment_line
    }

    existing_patient_data = df[df['treatment_line'] == treatment_line]
    existing_patient_data.set_index('patient_id', inplace=True)

    if st.button('Predict'):
        predicted_response = predict_drug_response(new_patient_data, scaler, existing_patient_data, networks, trained_models)
        st.success(f'Predicted Drug Response: {predicted_response}')

if __name__ == '__main__':
    main()
