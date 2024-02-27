import streamlit as st
import pandas as pd
import warnings

def main():
    st.set_page_config(layout="wide")  # Set the layout to wide for full-screen display
    st.title("PatStrat: A Framework for Embedding Multi-Modal Similarity Network for Patient Stratification")
    
    # Initialize a list to store uploaded files
    uploaded_files = []

    # Handle tab selection
    view_uploaded_files_tab(uploaded_files)


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
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file is not None:
            # Display file details
            with columns[i % int(num_files)]:
                # Display file content
                if uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file, index_col=0)
                    st.write(uploaded_file.name)
                    st.write(df)
                else:
                    st.write("File type not supported. Please upload a CSV file.")

def preprocess_files(uploaded_files):
    for file in uploaded_files:
        processed = preprocess_file(file)
        
#def preprocess_file(file):
    


if __name__ == "__main__":
    main()
