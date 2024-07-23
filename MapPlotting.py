import streamlit as st
import pandas as pd
from geopy.distance import geodesic
import pydeck as pdk

# Function to calculate the distance between two geographic coordinates
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).kilometers

def main():
    st.title('Geographic Distance Calculator for Clinical Trials')
    
    # Input for user's geographic coordinates
    st.subheader('Enter your geographic coordinates')
    user_lat = st.number_input('Latitude', value=39.6685, format="%.5f")
    user_lon = st.number_input('Longitude', value=-75.7048, format="%.5f")
    user_coords = (user_lat, user_lon)
    
    # Loading the data from the provided Excel file
    file_path = '20K_trials_geopoints.xlsx'
    df = pd.read_excel(file_path)
    
    # Ensuring geoPoint column is properly converted from string to list of floats
    df['geoPoint'] = df['geoPoint'].apply(eval)  # Converting string to list

    # Calculating distances and adding a new column
    df['distance'] = df['geoPoint'].apply(lambda x: calculate_distance(user_coords, tuple(x)))
    
    # Saving the dataframe to a new Excel file with distances
    output_file_1 = '20K_trials_with_distance.xlsx'
    df.to_excel(output_file_1, index=False)
    st.success(f'Excel file "{output_file_1}" with distance column has been generated.')

    # Input for proximity
    proximity = st.number_input('Enter the threshold distance(km):', min_value=0.0, step=1.0)
    
    # Filtering rows based on proximity
    filtered_df = df[df['distance'] <= proximity]
    
    # Keeping only the row with the smallest distance for each unique id
    final_df = filtered_df.loc[filtered_df.groupby('id')['distance'].idxmin()]

    # Saving the final dataframe to a new Excel file
    output_file_2 = '20K_distance_filtered_trials.xlsx'
    final_df.to_excel(output_file_2, index=False)
    st.success(f'Filtered Excel file "{output_file_2}" with unique trials has been generated.')

    # Button to view the trials
    if st.button('View'):
        st.write('Filtered Clinical Trials:')
        st.dataframe(final_df)
        st.write(f'Number of trials displayed: {len(final_df)}')

        # Ensure geoPoint is in list format and split into latitude and longitude
        final_df['geoPoint'] = final_df['geoPoint'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        final_df[['latitude', 'longitude']] = pd.DataFrame(final_df['geoPoint'].tolist(), index=final_df.index)

        # Adding user's coordinates to the dataframe for visualization
        user_data = pd.DataFrame({
            'latitude': [user_lat],
            'longitude': [user_lon],
            'id': ['User Location'],
            'location': ['User Location']
        })

        # Map settings for trials
        map_layer = pdk.Layer(
            'ScatterplotLayer',
            data=final_df,
            get_position='[longitude, latitude]',
            get_radius=10000,
            get_color=[255, 0, 0],
            pickable=True,
            tooltip=True
        )

        # Map settings for user's location
        user_layer = pdk.Layer(
            'ScatterplotLayer',
            data=user_data,
            get_position='[longitude, latitude]',
            get_radius=10000,
            get_color=[0, 0, 255],
            pickable=True
        )

        # Defining the text layer for labels
        text_layer = pdk.Layer(
            'TextLayer',
            data=final_df,
            get_position='[longitude, latitude]',
            get_text='locations',
            get_color=[0, 0, 0, 200],
            get_size=16,
            get_alignment_baseline='bottom'
        )

        view_state = pdk.ViewState(
            latitude=final_df['latitude'].mean(),
            longitude=final_df['longitude'].mean(),
            zoom=4,
            pitch=50
        )

        # Adding tooltip
        tooltip = {
            "html": "<b>Location:</b> {location}<br/><b>ID:</b> {id}<br/><b>Distance:</b> {distance} km",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }

        # Rendering the map
        st.pydeck_chart(pdk.Deck(
            initial_view_state=view_state,
            layers=[map_layer, user_layer, text_layer],
            tooltip=tooltip
        ))

if __name__ == '__main__':
    main()
