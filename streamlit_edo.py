# streamlit application for World Cup

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

# import datasets
df_matches = pd.read_csv("WorldCupMatches.csv")
df_players = pd.read_csv("WorldCupPlayers.csv")
df_world_cups = pd.read_csv("WorldCups.csv")


# run the app
# cd C:\Users\Edoardo\Documents\GitHub\DataVis_2022_2023
# streamlit run streamlit_edo.py

st.title("World Cup Data Visualization")

# ---------------------- SIDEBAR ----------------------
st.sidebar.subheader('Sections')
if st.sidebar.checkbox('Dataset'):
    st.header('World Cup Dataset')
    st.subheader('Matches')
    st.write(df_matches)
    st.subheader('Players')
    st.write(df_players)
    st.subheader('World Cups')
    st.write(df_world_cups)

# ------------------------------------------ World Cup Map ------------------------------------------
if st.sidebar.checkbox('World Cup Map'):
    st.header("World Cup Map")
    # print the winners of each world cup by counting the number of times each country appears in the winner column
    df_winner_cup = df_world_cups["Winner"].value_counts()
    # print(df_winner_cup)
    # add column names
    df_winner_cup = df_winner_cup.reset_index()
    df_winner_cup.columns = ["country", "wins"]
    # print(df_winner_cup)
    # Load world shapefile data
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Merge DataFrame with world shapefile data
    merged = world.merge(df_winner_cup, left_on='name', right_on='country', how='left')
    # Fill NaN values in the 'number of winner' column with 0
    merged['wins'] = merged['wins'].fillna(0)
    # Define color scheme
    color_map = 'YlOrBr'

    # Create a folium map centered on the world
    world_map = folium.Map(tiles='cartodbpositron')

    # Add choropleth layer to the map
    folium.Choropleth(
        geo_data=merged,
        data=merged,
        columns=['iso_a3', 'wins'],
        key_on='feature.properties.iso_a3',
        fill_color=color_map,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Number of World Cups won',
        highlight=True,
        nan_fill_color='white'  # Set the fill color for NaN (no winners) to white
    ).add_to(world_map)

    # Add tooltips to the map
    tooltip = folium.GeoJsonTooltip(fields=['name', 'wins'], aliases=['Country:', 'Number of World Cups won:'])
    folium.GeoJson(
        merged,
        tooltip=tooltip
    ).add_to(world_map)

    # Convert Folium map to HTML
    folium_map_html = world_map.get_root().render()

    # Display the map in Streamlit
    st.components.v1.html(folium_map_html, height=500)