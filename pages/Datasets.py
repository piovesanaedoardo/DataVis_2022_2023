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

def run():

    # import datasets
    df_matches = pd.read_csv("WorldCupMatches.csv")
    df_players = pd.read_csv("WorldCupPlayers.csv")
    df_world_cups = pd.read_csv("WorldCups.csv")

    # remove rows with missing values
    df_matches = df_matches.dropna()
    df_players = df_players.dropna()
    df_world_cups = df_world_cups.dropna()

    st.title("World Cup Data Visualization")

    # ---------------------- SIDEBAR ----------------------
    st.sidebar.subheader('Datasets')
    st.header('World Cup Datasets')

    if st.sidebar.checkbox('🏆 World Cups Dataset'):
        st.subheader('World Cups')
        st.write(df_world_cups)

    if st.sidebar.checkbox('🆚 Matches Dataset'):
        st.subheader('Matches')
        st.write(df_matches)

    if st.sidebar.checkbox('👨‍👨‍👦 Players Dataset'):
        st.subheader('Players')
        st.write(df_players)

