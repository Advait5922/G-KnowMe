import streamlit as st
import pandas as pd
import hashlib
import os
import shutil
import pickle
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq
import qdrant_client

nest_asyncio.apply()

st.title("Clinical Trials Eligibility Checker")

# Loading the data from the CSV file
file_path = 'ClinicalTrialsDataCSV.csv'
df = pd.read_csv(file_path)

# Handling missing values and ensure columns are strings
df['inclusion_criteria'] = df['inclusion_criteria'].fillna('').astype(str)
df['exclusion_criteria'] = df['exclusion_criteria'].fillna('').astype(str)

# Displaying the complete dataframe initially
st.subheader("Complete Clinical Trials Data")
st.write(df)

# Function to check age eligibility
def check_age_eligibility(user_age, trial_age):
    # Extracting the minimum age from the trial_age string
    age_part = trial_age.split()[0]
    if age_part.isdigit():
        min_age = int(age_part)
    else:
        if "(CHILD , ADULT , OLDER_ADULT)" in trial_age:
            min_age = 0
        else:
            return False  # If age information is not clear, exclude the trial

    return user_age >= min_age

# Function to write text to a temporary file
def write_temp_file(content, file_prefix):
    temp_file_path = f'./temp_files/{file_prefix}_{hashlib.md5(content.encode()).hexdigest()}.txt'
    with open(temp_file_path, 'w', encoding='utf-8') as temp_file:  # Ensure UTF-8 encoding
        temp_file.write(content)
    return temp_file_path

# Function to check exclusion criteria using LLM
def check_exclusion_criteria(trial_exclusion_criteria, comorb_list):
    LLAMAPARSE_API_KEY = st.secrets["LLAMAPARSE_API_KEY"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

    # Writing the exclusion criteria to a temporary file
    temp_file_path = write_temp_file(trial_exclusion_criteria, 'exclusion_criteria')

    # Parsing the data using LlamaParse
    llama_parse = LlamaParse(api_key=LLAMAPARSE_API_KEY, result_type="markdown")
    documents = llama_parse.load_data([temp_file_path])

    # Setting up embedding model
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
    from llama_index.core import Settings
    Settings.embed_model = embed_model

    # Setting up LLM model
    llm = Groq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY)
    Settings.llm = llm

    # Setting up Qdrant client and vector store
    client = qdrant_client.QdrantClient(api_key=QDRANT_API_KEY, url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name='qdrant_rag')
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, show_progress=True)
    index.storage_context.persist()

    # Creating a query engine for the index
    query_engine = index.as_query_engine()

    # Formulating the exclusion criteria query
    query = f'''If there is any word from the list: [{comorb_list}] present in the description: [{trial_exclusion_criteria}], return "No". Else say "Yes".'''

    # Querying the engine
    response = query_engine.query(query)
    # Writing the response to a temporary file
    response_file_path = write_temp_file(response.response.strip(), 'response')
    return response_file_path

# Function to clean up temporary folders and files
def cleanup_temp_files():
    data_dir = "./temp_files"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

# Sidebar for user inputs
st.sidebar.header("Filter Clinical Trials")

# Dropdown for Gender
gender_options = df['gender'].unique().tolist()
selected_gender = st.sidebar.selectbox("Select Gender", gender_options)

# Input for Age
input_age = st.sidebar.number_input("Enter Age", min_value=0, max_value=120, step=1)

# Dropdown for Comorbidities
comorbidities_options = [
    'None', 'Unknown', 'Diabetes mellitus', 'Thyroid diseases', 'Autoimmune disorders', 
    'Hypertension', 'HIV infection', 'HBV infection', 'HCV infection', 'Tuberculosis', 
    'Disorder affecting GI absorption', 'Liver disease', 'Respiratory infection', 
    'Interstitial lung disease', 'Congestive heart failure', 'Cardiac arrhythmia', 
    'Unstable angina pectoris', 'Myocardial infarction', 'Superior vena cava syndrome', 
    'Cardiomyopathy', 'Major seizure disorder', 'Unstable spinal cord compression', 
    'Psychiatric illness affecting social situations', 'Cognitive impairment', 
    'Neuromuscular disorders', 'Bleeding tendency', 'Inflammatory disorders', 
    'Immunodeficiency', 'Transplant Recipient', 'Allergic reactions to drugs and humanized antibodies', 
    'Substance abuse disorders', 'Not in a state to give consent', 'Other'
]
selected_comorbidities = st.sidebar.multiselect("Select Comorbidity", comorbidities_options)

if st.sidebar.button("Filter Clinical Trials"):
    # Filtering data based on user inputs
    filtered_df = df.copy()

    # Filtering by Gender
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['gender'] == selected_gender]

    # Filtering by Age
    filtered_df['age_eligibility'] = filtered_df['age'].apply(lambda x: check_age_eligibility(input_age, x))
    filtered_df = filtered_df[filtered_df['age_eligibility']]

    # Displaying the filtered dataframe
    st.subheader("Age and Gender filtered Clinical Trials Data")
    st.write(filtered_df)

    # Saving the filtered dataframe
    filtered_df.to_csv("filtered_clinical_trials.csv", index=False)

    # Starting exclusion criteria evaluation
    exclusion_info_placeholder = st.empty()  # Placeholder for exclusion info message
    exclusion_df_placeholder = st.empty()    # Placeholder for exclusion dataframe
    exclusion_info_placeholder.info("Starting exclusion criteria evaluation...")

    # Creating temp_files directory if it doesn't exist
    if not os.path.exists("./temp_files"):
        os.makedirs("./temp_files")

    # Evaluating exclusion criteria
    exclusion_eligible_trials = []
    for index, row in filtered_df.iterrows():
        trial_id = row['nct_id']
        trial_exclusion_criteria = row['exclusion_criteria']

        # Cleaning up temporary files before processing each trial
        cleanup_temp_files()

        # Updating message indicating the trial under evaluation
        exclusion_info_placeholder.info(f"Evaluating exclusion criteria for trial ID: {trial_id}...")

        response_file_path = check_exclusion_criteria(trial_exclusion_criteria, selected_comorbidities)

        # Reading the response from the temporary file
        with open(response_file_path, 'r', encoding='utf-8') as response_file:
            exclusion_eligibility = response_file.read().strip()

        # Deleting the temporary response file
        os.remove(response_file_path)

        if exclusion_eligibility == 'Yes.':
            row['eligibility'] = 'Eligible'
            exclusion_eligible_trials.append(row)
        
        # Updating the dataframe after exclusion criteria evaluation
        exclusion_eligible_df = pd.DataFrame(exclusion_eligible_trials)
        exclusion_df_placeholder.write(exclusion_eligible_df)

    # Saving the final dataframe
    exclusion_eligible_df.to_csv("final_eligible_trials.csv", index=False)

    # Displaying the final dataframe with eligibility column
    st.subheader("Final Eligible Clinical Trials")
    st.write(exclusion_eligible_df)
