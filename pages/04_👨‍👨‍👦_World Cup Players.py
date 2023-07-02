# libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

# import datasets
df_matches = pd.read_csv("WorldCupMatches.csv")
df_players = pd.read_csv("WorldCupPlayers.csv")
df_world_cups = pd.read_csv("WorldCups.csv")

st.title("World Cup Data Visualization")
st.header("World Cups Players")