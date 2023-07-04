# libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import base64

def run():

    # --- IMPORT DATASETS --- #
    # import datasets
    df_matches = pd.read_csv("WorldCupMatches.csv")
    df_players = pd.read_csv("WorldCupPlayers.csv")
    df_world_cups = pd.read_csv("WorldCups.csv")
    
    st.title("World Cup Data Visualization")
    st.header("World Cups Matches")

    # --- MODIFY DATASETS --- #
    # Calculate the total goals scored by each team in each match
    df_matches['Home Team Goals'] = df_matches['Home Team Goals'].fillna(0).astype(int)
    df_matches['Away Team Goals'] = df_matches['Away Team Goals'].fillna(0).astype(int)
    df_matches['Total Home Goals'] = df_matches['Home Team Goals']
    df_matches['Total Away Goals'] = df_matches['Away Team Goals']

    # --- CREATE GOALS DATASET --- #
    # Reshape the data to have one row per team per match
    df_home = df_matches[['Year', 'Home Team Name', 'Total Home Goals']].rename(columns={'Home Team Name': 'Team', 'Total Home Goals': 'Goals'})
    df_away = df_matches[['Year', 'Away Team Name', 'Total Away Goals']].rename(columns={'Away Team Name': 'Team', 'Total Away Goals': 'Goals'})
    df_goals = pd.concat([df_home, df_away])

    # Calculate the total goals scored by each team in each year
    df_goals = df_goals.groupby(['Year', 'Team'], as_index=False).sum()

    # Pivot the data to have one column per team
    df_goals = df_goals.pivot(index='Year', columns='Team', values='Goals').fillna(0).astype(int)

    # Calculate the cumulative sum of goals for each team
    df_goals = df_goals.cumsum()

    # Reset the index to have Year as a column
    df_goals = df_goals.reset_index()

    # Melt the data to have one row per team per year
    df_goals = df_goals.melt(id_vars='Year', var_name='Team', value_name='Goals')

    st.write(df_goals)
    '''
    # --- VS_1) CREATE A RACING CHART --- #
    # create a racing bar chart to visualize the total goals scored by each team in each year

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.animation as animation
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Set the style of the chart
    plt.style.use('seaborn-darkgrid')

    # Set the number of teams to display
    n_teams = 10

    # Set the duration of the animation in milliseconds
    duration = 900

    # Create the figure and axes objects
    fig, ax = plt.subplots(figsize=(15, 8))

    # Function to draw the chart for a given year
    def draw_chart(year):
        # Filter the data for the given year
        df_year = df_goals[df_goals['Year'].eq(year)].sort_values(by='Goals', ascending=False).head(n_teams)
        df_year = df_year[::-1]  # reverse the order for plotting
        
        # Clear the previous chart
        ax.clear()
        
        # Check if there is data for the given year
        if not df_year.empty:
            # Set the x and y limits
            ax.set_xlim(0, df_year['Goals'].max())
            ax.set_ylim(-0.5, n_teams - 0.5)
            
            # Set the x and y ticks
            ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
            ax.yaxis.set_ticks(range(n_teams))
            ax.yaxis.set_ticklabels(df_year['Team'])
            
            # Set the title and labels
            ax.set_title(f'Total Goals Scored by Each Team: {year}', fontsize=24)
            ax.set_xlabel('Goals', fontsize=18)
            ax.set_ylabel('Team', fontsize=18)
            
            # Plot the bars
            bars = ax.barh(df_year['Team'], df_year['Goals'], height=0.8, color='steelblue')
            
            # Add value labels to the bars
            for bar in bars:
                width = bar.get_width()
                label = f'{width:.0f}'
                x = width + 5
                y = bar.get_y() + bar.get_height() / 2
                ax.text(x, y, label, ha='left', va='center', fontsize=14)


    # Create the animation object
    start_year = int(df_goals['Year'].min())
    end_year = int(df_goals['Year'].max())
    animator = FuncAnimation(fig, draw_chart, frames=range(start_year, end_year + 1), interval=duration)

    # Save animation as gif file 
    writer = PillowWriter(fps=60) 
    animator.save("animation.gif", writer=writer)

    # Display the animation
    """### gif from local file"""
    file_ = open("animation.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width="750" height="400" style="animation-duration: 5s;">',
        unsafe_allow_html=True
    )


    # ------- nuovo grafico con streamlit martin -------- #
    #codice ...
    # @martin sarebbe interessante mostrare:
    # - la partita con più goal della storia
    # - la partita con più pubblico della storia
    # - la squadra con più vittorie nella storia
    # - la squadra con più sconfitte nella storia
    '''
    st.write('sss')