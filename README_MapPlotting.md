# README

## Geographic Distance Calculator for Clinical Trials

This Streamlit application allows users to input their geographic coordinates and calculates the distance between their location and various clinical trial locations. The application filters the trials based on a specified proximity and provides a visual representation of the trials on a map.

### Features
1. **User Input**: Users can input their geographic coordinates (latitude and longitude).
2. **Data Loading**: Loads clinical trial data from an Excel file.
3. **Distance Calculation**: Calculates the distance between the user's coordinates and the trial locations.
4. **Proximity Filtering**: Filters trials based on the specified proximity distance.
5. **Excel Output**: Saves filtered data into Excel files.
6. **Data Visualization**: Displays the filtered clinical trials on a map using Pydeck.

### Input Fields
- **Latitude**: Number input for latitude.
- **Longitude**: Number input for longitude.
- **Proximity**: Number input for threshold distance (km).

### Functions
- **calculate_distance**: Calculates the geographic distance between two coordinates using the geodesic method.

### Process Workflow
1. **Input User Coordinates**: Users input their geographic coordinates.
2. **Data Loading**: 
    - Loads clinical trial data from the `20K_trials_geopoints.xlsx` file.
    - Converts geographic coordinates from strings to lists of floats.
3. **Distance Calculation**:
    - Calculates distances between the user's coordinates and trial locations.
    - Saves the dataframe with distances to `20K_trials_with_distance.xlsx`.
4. **Proximity Filtering**:
    - Filters rows based on the specified proximity distance.
    - Keeps only the row with the smallest distance for each unique trial ID.
    - Saves the filtered data to `20K_distance_filtered_trials.xlsx`.
5. **Data Visualization**:
    - Displays the filtered clinical trials on a map.
    - Adds a layer for the user's location.
    - Provides tooltips with trial information on the map.

### Output
- **20K_trials_with_distance.xlsx**: Contains the data with calculated distances.
- **20K_distance_filtered_trials.xlsx**: Contains the filtered data within the specified proximity.

### Libraries Used
- `streamlit`
- `pandas`
- `geopy.distance` (geodesic)
- `pydeck`

### Running the Application
To run the application, execute the script using Streamlit:

```bash
streamlit run your_script_name.py
