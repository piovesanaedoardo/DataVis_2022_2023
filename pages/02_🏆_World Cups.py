# libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
# mapping the winner country in a map
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import folium
from branca.colormap import linear
# interactive histogram
import plotly.express as px
import plotly.graph_objects as go

# import datasets
df_matches = pd.read_csv("WorldCupMatches.csv")
df_players = pd.read_csv("WorldCupPlayers.csv")
df_world_cups = pd.read_csv("WorldCups.csv")

st.title("World Cup Data Visualization")
st.header("World Cups")

# ---------------------- SIDEBAR ----------------------
if st.sidebar.checkbox('Show Dataset'):
    st.subheader('World Cups')
    st.write(df_world_cups)

print(df_world_cups)

# ---------------------- WORLD MAP ----------------------
st.subheader("World Map")
# Load the dataset
df_worldmap_cup = pd.DataFrame({
    'Country': ['Brazil', 'Italy', 'Germany FR', 'Uruguay', 'Argentina', 'England', 'France', 'Spain', 'Germany', 'Netherlands',
                'Czechoslovakia', 'Hungary', 'Sweden', 'Poland', 'USA', 'Austria', 'Chile', 'Portugal', 'Croatia', 'Turkey',
                'Yugoslavia', 'Soviet Union', 'Belgium', 'Bulgaria', 'Korea Republic'],
    'Winner': [5.0, 4.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Second': [2.0, 2.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 3.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Third': [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Fourth': [2.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0]
})
# Load world shapefile data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# Set 'Country' column as the index
df_worldmap_cup.set_index('Country', inplace=True)
# Reshape the dataframe using melt function
df_worldmap_cup = df_worldmap_cup.reset_index().melt(id_vars='Country', var_name='Placement', value_name='Count')
# Merge the world shapefile data with the world cup data
merged = world.merge(df_worldmap_cup, left_on='name', right_on='Country', how='left')
# Fill NaN values in the 'Count' column with 0
merged['Count'] = merged['Count'].fillna(0)
# Define color scheme
color_map = 'YlOrBr'
# Create a folium map centered on the world
world_map = folium.Map(tiles='cartodbpositron')
# Create a function to update the map based on the selected placement
def update_map(selected_placement):
    # Filter the merged dataframe based on the selected placement
    filtered_data = merged[merged['Placement'] == selected_placement]  
    # Create choropleth layer for the filtered data
    choropleth = folium.Choropleth(
        geo_data=filtered_data,
        data=filtered_data,
        columns=['iso_a3', 'Count'],
        key_on='feature.properties.iso_a3',
        fill_color=color_map,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='World Cup Placements',
        highlight=True,
        nan_fill_color='white',
        bins=range(int(filtered_data['Count'].max()) + 2)  # Set the legend bins to integer values from 0 to max+1
    )
    # Add tooltips to the map
    tooltip = folium.GeoJsonTooltip(fields=['name', 'Count'], aliases=['Country:', 'Count:'])
    choropleth.geojson.add_child(tooltip)
    # Add the choropleth layer to the map
    choropleth.add_to(world_map)
# Get the available placement options
placements = df_worldmap_cup['Placement'].unique()
# Create a dropdown menu to select the placement
selected_placement = st.selectbox('Select Placement:', placements)
# Update the map based on the selected placement
update_map(selected_placement)
# Display the map in Streamlit
st.components.v1.html(world_map._repr_html_(), height=500)

# ---------------------- HISTORY TREND ----------------------
st.subheader("History Trend")
# ------------- GOALS SCORED PER YEAR -------------
# histogram of goals scored per year
fig = px.bar(x=df_world_cups["Year"], y=df_world_cups["GoalsScored"], 
            title="Goals Scored per Year",
            labels={'x': 'Year', 'y': 'Number of Goals'}, 
            height=400)
# Set x-ticks every 4 years from 1930 to 2014
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=np.arange(1930, 2015, 4),
        ticktext=np.arange(1930, 2015, 4)
    )
)
st.plotly_chart(fig)

# ------------- MATCHES PLAYED PER YEAR -------------
# histogram of matches played per year
fig = px.bar(x=df_world_cups["Year"], y=df_world_cups["MatchesPlayed"],
                title="Matches Played per Year",
                labels={'x': 'Year', 'y': 'Number of Matches'},
                orientation='v',
                height=400)
# Set x-ticks every 4 years from 1930 to 2014
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=np.arange(1930, 2015, 4),
        ticktext=np.arange(1930, 2015, 4)
    )
)
st.plotly_chart(fig)

# ------------- GOALS SCORED BY MATCHES PLAYED IN EACH YEAR -------------
# histogram of goals scored by matches played in each year
fig = px.bar(x=df_world_cups["Year"], y=df_world_cups["GoalsScored"]/df_world_cups["MatchesPlayed"],
                title="Goals Scored by Matches Played in each Year",
                labels={'x': 'Year', 'y': 'Goals per Match'},
                height=400)
# Set x-ticks every 4 years from 1930 to 2014
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=np.arange(1930, 2015, 4),
        ticktext=np.arange(1930, 2015, 4)
    )
)
st.plotly_chart(fig)