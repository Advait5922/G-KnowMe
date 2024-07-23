# README

## Patient Condition Input Form

This Streamlit application allows users to input patient conditions and filters clinical trial data based on the provided inputs. The application then processes the data to calculate distances and similarities and saves the filtered data into Excel files.

### Features
1. **Input Form**: Users can input patient details such as age, gender, status, phase, primary tumor type, sub-tumor type, latitude, and longitude.
2. **Data Filtering**: Filters clinical trial data based on the provided inputs:
    - Age and Gender
    - Recruitment Status
    - Phases
3. **Distance Calculation**: Calculates the distance between the user's coordinates and the trial locations.
4. **Ollama Embeddings**: Calculates similarity scores using Ollama embeddings for tumor conditions.
5. **Excel Output**: Saves filtered and processed data into Excel files.

### Input Fields
- **Age**: Number input for age (0 to 120).
- **Gender**: Dropdown selection for gender (ALL, FEMALE, MALE).
- **Status**: Dropdown selection for recruitment status (RECRUITING, NOT_YET_RECRUITING).
- **Phase**: Multiselect for trial phases (EARLY_PHASE1, PHASE1, PHASE2, PHASE3, PHASE4).
- **Primary Tumor**: Dropdown selection for primary tumor type.
- **Sub-Tumor**: Dropdown selection for sub-tumor type, based on selected primary tumor.
- **Latitude**: Number input for latitude.
- **Longitude**: Number input for longitude.
- **Proximity**: Number input for threshold distance (km).

### Functions
- **get_ollama_embedding**: Fetches embedding for a given text using the Ollama API.
- **calculate_ollama_similarity**: Calculates similarity scores between the query and corpus using cosine similarity of Ollama embeddings.
- **calculate_distance**: Calculates the geographic distance between two coordinates using the geodesic method.

### Process Workflow
1. **Input Patient Data**: Users input patient conditions and coordinates.
2. **Sort Button Click**: 
    - Loads clinical trial data from an Excel file.
    - Filters data based on age, gender, status, and phases.
    - Converts geographic coordinates from strings to lists.
    - Calculates distances between the user's coordinates and trial locations.
    - Saves the filtered data with distances to an Excel file.
    - Filters data further based on proximity.
    - Saves the final filtered data to another Excel file.
3. **Ollama Embeddings**:
    - Calculates similarity scores for tumor conditions using Ollama embeddings.
    - Saves the similarity scores to a new Excel file.
    - Processes the data and appends similarity scores for each condition.
    - Saves the processed data to a final Excel file.

### Output
- **json_to_excel_distances.xlsx**: Contains filtered data with calculated distances.
- **json_to_excel_distances_proximity.xlsx**: Contains filtered data within the specified proximity.
- **Ollama_{patient_condition}.xlsx**: Contains similarity scores for each condition processed from the proximity filtered data.

### Libraries Used
- `streamlit`
- `pandas`
- `numpy`
- `sklearn.metrics.pairwise` (cosine_similarity)
- `ollama`
- `geopy.distance` (geodesic)
- `pydeck`
