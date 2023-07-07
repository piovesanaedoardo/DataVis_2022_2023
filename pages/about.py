# libraries
import streamlit as st
def run():
    st.title("World Cup Data Visualization")

    st.header('Brief Description')
    st.markdown('''
    The FIFA World Cup is a globally celebrated sporting event with a rich history and immense popularity. 
    Our motivation for choosing this particular task/application field is twofold. 
    Firstly, the World Cup dataset provides a diverse range of information, including match results, player statistics, 
    and tournament details, which can be effectively visualized to gain insights into the game's evolution over time. 
    Secondly, the World Cup captures the attention of millions of fans worldwide, making it an exciting and relatable subject for data visualization. 
    The data is taken from [Kaggle](https://www.kaggle.com/abecklas/fifa-world-cup).
                
    This project aims to visualize the World Cup data in a way that is both informative and engaging. We have divided the project into three main sections::
    - **World Cup Overview**: This section provides an overview of the World Cup's history and statistics
    - **World Cup Players**: This section allows users to explore the players' statistics
    - **World Cup Matches**: This section allows users to explore the matches' statistics
    ''')

    # add image
    st.image("World_Cup_Trophy.png", width=700)

