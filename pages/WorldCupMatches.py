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
    
    ''''
    #Clean the variables internal strings
    import re

    # Define the cleaning function
    def clean_value(value):
        value = str(value)  # Convert to string if not already
        
        # Remove leading/trailing whitespaces
        value = value.strip()
        
        # Remove dashes
        #value = value.replace('-', '')
        
        # Remove non-alphanumeric characters except spaces
        value = re.sub(r'[^a-zA-Z0-9\s]', '', value)
        
        return value

    # Apply the cleaning function to specific columns
    columns_to_clean = ['Home Team Name', 'Away Team Name', 'Stadium', 'City'] #, 'Referee', 'Assistant 1', 'Assistant 2']
    df_matches[columns_to_clean] = df_matches[columns_to_clean].applymap(clean_value)
    '''

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

    # --- VS_1) la partita con più goal della storia / la partita con più pubblico della storia --- #
    '''
    # Create a DataFrame with the required information
    df_matches['Total Goals'] = df_matches['Home Team Goals'] + df_matches['Away Team Goals']
    max_goals_match = df_matches.loc[df_matches['Total Goals'].idxmax()]
    max_attendance_match = df_matches.loc[df_matches['Attendance'].idxmax()]
    combined_table_max_goals_attendance = pd.concat([max_goals_match, max_attendance_match], axis=1)
    combined_table_max_goals_attendance.columns = ["Match with the Most Goals in History", "Match with the Most Attendance in History"]

    # Display the combined table
    st.header("Matches With Most Goals and Attendance in WC History")
    st.write(combined_table_max_goals_attendance)
    '''
    
    # --- VS_2) la squadra con più vittorie nella storia / la squadra con più sconfitte nella storia --- #

    # Team with the most wins
    wins = df_matches[df_matches['Home Team Goals'] > df_matches['Away Team Goals']]['Home Team Name'].append(
        df_matches[df_matches['Away Team Goals'] > df_matches['Home Team Goals']]['Away Team Name'])

    most_wins_team = wins.value_counts().idxmax()

    # Team with the most defeats
    defeats = df_matches[df_matches['Home Team Goals'] < df_matches['Away Team Goals']]['Home Team Name'].append(
            df_matches[df_matches['Away Team Goals'] < df_matches['Home Team Goals']]['Away Team Name'])

    most_defeats_team = defeats.value_counts().idxmax()

    # Create a DataFrame with the required information
    combined_table_max_wins_losses = pd.DataFrame({
        "Team with the Most Wins in History": [most_wins_team],
        "Team with the Most Defeats in History": [most_defeats_team]})

    # Display the combined table
    st.header("Team With Most Wins and Losses in WC History")
    st.table(combined_table_max_wins_losses)

    
    # --- VS_3.1) la squadra con più vittorie nella storia / la squadra con più sconfitte nella storia 
    #           per numero di partite giocate - table --- #

    # Create a function to determine the winner and loser
    def get_results(row):
        if row['Home Team Goals'] > row['Away Team Goals']:
            return pd.Series([row['Home Team Name'], row['Away Team Name']])
        elif row['Home Team Goals'] < row['Away Team Goals']:
            return pd.Series([row['Away Team Name'], row['Home Team Name']])
        else:
            return pd.Series([np.nan, np.nan])

    # Apply the function to the DataFrame
    df_matches[['Winner', 'Loser']] = df_matches.apply(get_results, axis=1)

    # Count the number of matches, wins and losses for each team
    matches_played = df_matches['Home Team Name'].value_counts() + df_matches['Away Team Name'].value_counts()
    wins = df_matches['Winner'].value_counts().reindex(matches_played.index, fill_value=0)
    losses = df_matches['Loser'].value_counts().reindex(matches_played.index, fill_value=0)

    # Calculate the sum of wins and losses
    total_wins_losses = wins + losses

    # Replace NaN values in matches_played with the sum of wins and losses
    matches_played = matches_played.fillna(total_wins_losses)

    # Calculate the win-loss ratio
    win_loss_ratio = wins / losses.replace(0, 1)  # replace 0 with 1 in losses to avoid division by zero

    # Calculate wins/matches and losses/matches
    win_match_ratio = wins / matches_played
    loss_match_ratio = losses / matches_played

    # Combine the statistics into a single DataFrame
    team_stats = pd.concat([matches_played, wins, losses, win_loss_ratio, win_match_ratio, loss_match_ratio], axis=1)
    team_stats.columns = ['Matches Played', 'Wins', 'Losses', 'Win-Loss Ratio', 'Win-Match Ratio', 'Loss-Match Ratio']

    # Display the table in Streamlit
    st.header("Summary Statistics by Team: Matches Played, Wins, Losses, Win-Loss Ratio, Win-Match Ratio, Loss-Match Ratiovin WC History")
    st.table(team_stats)

    
    # --- VS_3) Team Statistics by Year --- #
    def get_results(row):
        if row['Home Team Goals'] > row['Away Team Goals']:
            return pd.Series([row['Year'], row['Home Team Name'], row['Away Team Name']])
        elif row['Home Team Goals'] < row['Away Team Goals']:
            return pd.Series([row['Year'], row['Away Team Name'], row['Home Team Name']])
        else:
            return pd.Series([np.nan, np.nan, np.nan])

    df_matches[['Year', 'Winner', 'Loser']] = df_matches.apply(get_results, axis=1)

    home_matches = df_matches.groupby(['Year', 'Home Team Name']).size()
    away_matches = df_matches.groupby(['Year', 'Away Team Name']).size()
    matches_played = (home_matches + away_matches).fillna(0)

    wins = df_matches.groupby(['Year', 'Winner']).size().reindex(matches_played.index, fill_value=0)
    losses = df_matches.groupby(['Year', 'Loser']).size().reindex(matches_played.index, fill_value=0)

    # Calculate the win-loss ratio
    win_loss_ratio = wins / losses.replace(0, 1)  # replace 0 with 1 in losses to avoid division by zero

    # Calculate wins/matches and losses/matches
    win_match_ratio = wins / matches_played
    loss_match_ratio = losses / matches_played

    # Combine the statistics into a single DataFrame
    team_stats_yearly = pd.concat([matches_played, wins, losses, win_loss_ratio, win_match_ratio, loss_match_ratio], axis=1)
    team_stats_yearly.columns = ['Matches Played', 'Wins', 'Losses', 'Win-Loss Ratio', 'Win-Match Ratio', 'Loss-Match Ratio']

    # Display the table in Streamlit
    st.header("Summary Statistics by Team and Year: Matches Played, Wins, Losses, Win-Loss Ratio, Win-Match Ratio, Loss-Match Ratio")
    st.table(team_stats_yearly)

    # --- VS_4) Line Chart for each country by year --- #
    countries = df_matches['Home Team Name'].unique()  # replace with your list of countries

    for country in countries:
        country_data = team_stats_yearly.xs(country, level=1)
        st.line_chart(country_data[['Wins', 'Losses']])
    