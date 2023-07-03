# libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import urllib.parse
import streamlit as st


def run():
    # import datasets
    df_matches = pd.read_csv("WorldCupMatches.csv")
    df_players = pd.read_csv("WorldCupPlayers.csv")
    df_world_cups = pd.read_csv("WorldCups.csv")

    st.title("World Cup Data Visualization")
    st.header("World Cups Players")
    #data cleaning
    df_merge = df_matches[['Year',"MatchID"]]
    merged_df = pd.merge(df_players, df_merge, on='MatchID', how='left')
    df_players = merged_df
    df_players['Player Name'] = df_players['Player Name'].str.normalize('NFKD')
    df_players.drop_duplicates(inplace=True)
    #split the event column
    df_players['Event'].fillna('', inplace=True)
    df_players['Goals'] = df_players['Event'].apply(lambda x: sum(1 for event in x.split() if event.startswith('G')))
    df_players['OwnGoals'] = df_players['Event'].apply(lambda x: sum(1 for event in x.split() if event.startswith('OG')))
    df_players['Penalties'] = df_players['Event'].apply(lambda x: sum(1 for event in x.split() if event.startswith('P')))
    df_players['PenaltiesMissed'] = df_players['Event'].apply(lambda x: sum(1 for event in x.split() if event.startswith('MP')))
    df_players['YellowCards'] = df_players['Event'].apply(lambda x: sum(1 for event in x.split() if event.startswith('Y')))
    df_players['RedCards'] = df_players['Event'].apply(lambda x: sum(1 for event in x.split() if event.startswith('R')))
    df_players['RedCardsByYellow'] = df_players['Event'].apply(lambda x: sum(1 for event in x.split() if event.startswith('SY')))
    df_players['SubstituteIn'] = df_players['Event'].apply(lambda x: sum(1 for event in x.split() if event.startswith('I')))
    df_players['SubstituteOut'] = df_players['Event'].apply(lambda x: sum(1 for event in x.split() if event.startswith('O')))
    df_players['Minute'] = df_players['Event'].str.extract('(\d+)').astype(float).fillna(0).astype(int)    
    df_players['Team Initials'] = df_players['Team Initials'].replace('FRG', 'GER')
    df_players['Team Initials'] = df_players['Team Initials'].replace('URS', 'RUS')
    df_players['Team Initials'] = df_players['Team Initials'].replace('TCH', 'CZE')

    #Barplot number of players by team

    team_counts = df_players.groupby('Team Initials')['Player Name'].count().reset_index()

    # Sorting teams based on the number of players
    team_counts = team_counts.sort_values('Player Name', ascending=False)

    # Creating the bar chart using Plotly Express
    fig = px.bar(team_counts, x='Team Initials', y='Player Name', title='Number of Players by Team', color='Team Initials')

    # Updating hover template to show 'Number of Players'
    fig.update_traces(hovertemplate='Number of Players: %{y}')

    # Displaying the interactive chart
    st.write(fig)