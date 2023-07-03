import streamlit as st
from pages import about, Datasets, WorldCups, WorldCupMatches, WorldCupPlayers

PAGES = {
    "About": about,
    "Datasets": Datasets,
    "World Cups": WorldCups,
    "World Cup Matches": WorldCupMatches,
    "World Cup Players": WorldCupPlayers
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page.run()

if "shared" not in st.session_state:
   st.session_state["shared"] = True

# to run streamlit app:
# cd C:\Users\Edoardo\Documents\GitHub\DataVis_2022_2023
# streamlit run main.py