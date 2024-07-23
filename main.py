import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from geopy.distance import geodesic
import pydeck as pdk

# Streamlit app
st.title("Patient Condition Input Form")

# Create columns for single row layout
col1, col2, col3, col4 = st.columns(4)
col5, col6, col7, col8 = st.columns(4)
# Age input
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, step=1)

# Gender input
with col2:
    gender = st.selectbox("Gender", options=["", "ALL", "FEMALE", "MALE"], index=0)

# Status input
with col3:
    status_options = ["RECRUITING", "NOT_YET_RECRUITING", ""]
    status = st.selectbox("Status", options=[""] + status_options, index=0)

# Phase input
with col4:
    phase_options = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]
    phase = st.multiselect("Phase", options=[""] + phase_options)

# Primary tumor input
with col5:
    primary_tumor_options = ["Adrenal Gland Cancer", "Ampullary Carcinoma"]
    primary_tumor = st.selectbox("Primary Tumor", options=[""] + primary_tumor_options, index=0)

# Sub-tumor input
with col6:
    sub_tumor_options = []
    if primary_tumor == "Adrenal Gland Cancer":
        sub_tumor_options = ["Adrenocortical Adenoma (ACA)", "Adrenocortical Carcinoma (ACC)", "Pheochromocytoma (PHC)"]
    elif primary_tumor == "Ampullary Carcinoma":
        sub_tumor_options = ["Intestinal Ampullary Carcinoma (IAMPCA)", "Mixed Ampullary Carcinoma (MAMPCA)", "Pancreatobiliary Ampullary Carcinoma (PAMPCA)"]

    sub_tumor = st.selectbox("Sub-Tumor Type", options=[""] + sub_tumor_options, index=0)

# Latitude input
with col7:
    user_lat = st.number_input('Latitude', value=38.98067, format="%.5f")

# Longitude input
with col8:
    user_lon = st.number_input('Longitude', value=-77.10026, format="%.5f")
       
# Input for proximity
proximity = st.number_input('Enter the threshold distance(km):', min_value=0.0, step=1.0)

# Function to get Ollama embedding
def get_ollama_embedding(text):
    try:
        response = ollama.embeddings(model='mxbai-embed-large', prompt=text)
        return response['embedding']
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        return None

# Function to calculate similarity using Ollama embeddings
def calculate_ollama_similarity(corpus, query):
    if not corpus:
        return []
    
    query_embedding = get_ollama_embedding(query)
    if query_embedding is None:
        return []
    
    scores = []
    for doc in corpus:
        doc_embedding = get_ollama_embedding(doc)
        if doc_embedding is not None:
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            scores.append(similarity)
        else:
            scores.append(0)
    
    return scores
 
user_coords = (user_lat, user_lon)
patient_condition = f"{primary_tumor}, {sub_tumor}"
if st.button('Sort'):

    # Loading trial data
    trial_data = pd.read_excel("json_to_excel2.xlsx")

    # Filtering data based on age and gender
    filtered_data = trial_data[(trial_data['minimumAge'] >= age) & (trial_data['gender'].isin([gender, "ALL"]))]
    st.write("Age and Gender sorted")
    #filtered_data = filtered_data[filtered_data['minimumAge'] >= age]
    
    # Filtering the dataframe based on the selected status
    if status:
        filtered_data = filtered_data[filtered_data['status'] == status]
    st.write("Status sorted")
    
    # Filtering based on Phase selection
    def filter_phases(row, selected_phases):
        phases = eval(row['phases'])  # Convert the string representation of list back to a list
        for sp in selected_phases:
            if sp in phases:
                return True
        return False

    if phase:
        filtered_data = filtered_data[filtered_data.apply(lambda row: filter_phases(row, phase), axis=1)]
    st.write("Phases sorted")
    
    # Function to calculate the distance between two geographic coordinates
    def calculate_distance(coord1, coord2):
        return geodesic(coord1, coord2).kilometers

    # Ensuring geoPoint column is properly converted from string to list of floats
    filtered_data['geoPoint'] = filtered_data['geoPoint'].apply(eval)  # Converting string to list

    # Calculating distances and adding a new column
    filtered_data['distance'] = filtered_data['geoPoint'].apply(lambda x: calculate_distance(user_coords, tuple(x)))
    
    # Saving the dataframe to a new Excel file with distances
    output_file_1 = 'json_to_excel_distances.xlsx'
    filtered_data.to_excel(output_file_1, index=False)
    st.success(f'Excel file "{output_file_1}" with distance column has been generated.')

    filtered_data = filtered_data[filtered_data['distance'] <= proximity]

    # Keeping only the row with the smallest distance for each unique id
    #filtered_data = filtered_data.loc[filtered_data.groupby('id')['distance'].idxmin()]

    # Saving the final dataframe to a new Excel file
    output_file_2 = 'json_to_excel_distances_proximity.xlsx'
    filtered_data.to_excel(output_file_2, index=False)
    st.success(f'Excel file "{output_file_2}" with distance within proximity has been generated.')
    
    # Processing similarity scores using Ollama Embeddings
    def process_tumor_data(patient_condition):
        df = pd.read_excel('json_to_excel_distances_proximity.xlsx')
    
        # Calculating similarity scores for each condition in the DataFrame
        df['Similarity Scores'] = df['conditions'].apply(lambda conditions: calculate_ollama_similarity(eval(conditions), patient_condition) if isinstance(conditions, str) and conditions.strip() else [])
    
        # Saving the similarity scores to a new Excel file
        output_file = f'Ollama_{patient_condition.replace(" ", "_")}.xlsx'
        df.to_excel(output_file, index=False)
    
        # Processing the data for the new Excel file
        input_df = pd.read_excel(output_file)
        processed_data = []
        for index, row in input_df.iterrows():
            id = row[0]
            gender = row[1]
            minimumAge = row[2]
            status = row[3]
            phases = row[4]
            conditions = eval(row[5]) if isinstance(row[5], str) and row[5].strip() else []
            geoPoint = row[6]
            distance = row[7]
            scores = eval(row[8]) if isinstance(row[8], str) and row[8].strip() else []
            for condition, score in zip(conditions, scores):
                processed_data.append([id, gender, minimumAge, status, phases, condition, geoPoint, distance, score])
            processed_data.append([None, None, None, None, None, None, None, None, None])  # Adding an empty row after each unique NCT ID
    
        # Saving the processed data to a new Excel file
        processed_df = pd.DataFrame(processed_data, columns=["id", "gender", "minimumAge", "status", "phases", "conditions", "geoPoint", "distance","Scores"])
        processed_output_file = f'Ollama_{patient_condition.replace(" ", "_")}.xlsx'
        processed_df.to_excel(processed_output_file, index=False)
        print(f"Data has been processed and saved to {processed_output_file}")
    
    process_tumor_data(patient_condition)
    