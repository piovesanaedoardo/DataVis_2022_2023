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
    #             per numero di partite giocate - table --- #

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
    team_stats_1 = pd.concat([matches_played, wins, losses, win_loss_ratio, win_match_ratio, loss_match_ratio], axis=1)
    team_stats_1.columns = ['Matches Played', 'Wins', 'Losses', 'Win-Loss Ratio', 'Win-Match Ratio', 'Loss-Match Ratio']

    # Display the table in Streamlit
    st.header("Summary Statistics by Team: Matches Played, Wins, Losses, Win-Loss Ratio, Win-Match Ratio, Loss-Match Ratiovin WC History")
    st.table(team_stats_1)


    # --- VS_3.2) la squadra con più vittorie nella storia / la squadra con più sconfitte nella storia 
    #             - table by year --- #
    
    import plotly.express as px

    # Create an empty DataFrame
    team_stats = pd.DataFrame()

    # For each year, count the number of matches, wins, and losses for each team
    for year in df_matches['Year'].unique():
        # Get all matches in this year
        matches_this_year = df_matches[df_matches['Year'] == year]

        # Count the matches, wins, and losses for each team
        home_teams = matches_this_year['Home Team Name'].value_counts().reset_index()
        away_teams = matches_this_year['Away Team Name'].value_counts().reset_index()

        home_wins = matches_this_year[matches_this_year['Home Team Goals'] > matches_this_year['Away Team Goals']]['Home Team Name'].value_counts().reset_index()
        away_wins = matches_this_year[matches_this_year['Away Team Goals'] > matches_this_year['Home Team Goals']]['Away Team Name'].value_counts().reset_index()

        home_losses = matches_this_year[matches_this_year['Home Team Goals'] < matches_this_year['Away Team Goals']]['Home Team Name'].value_counts().reset_index()
        away_losses = matches_this_year[matches_this_year['Away Team Goals'] < matches_this_year['Home Team Goals']]['Away Team Name'].value_counts().reset_index()

        # Merge the counts into one DataFrame
        teams = home_teams.merge(away_teams, how='outer', on='index').fillna(0)
        wins = home_wins.merge(away_wins, how='outer', on='index').fillna(0)
        losses = home_losses.merge(away_losses, how='outer', on='index').fillna(0)

        # Compute the total number of matches, wins, and losses
        teams['matches'] = teams['Home Team Name'] + teams['Away Team Name']
        wins['wins'] = wins['Home Team Name'] + wins['Away Team Name']
        losses['losses'] = losses['Home Team Name'] + losses['Away Team Name']

        # Merge all stats into one DataFrame and append it to the main DataFrame
        year_stats = teams.merge(wins, how='outer', on='index').merge(losses, how='outer', on='index')
        year_stats['year'] = year

        # Append the stats of this year to the main DataFrame
        team_stats = pd.concat([team_stats, year_stats])

    # Rename the columns
    team_stats.columns = ['Team', 'Home Matches', 'Away Matches', 'Matches', 'Home Wins', 'Away Wins', 'Wins', 'Home Losses', 'Away Losses', 'Losses', 'Year']

    # --- VS_3.2.1) linechart by year and country --- #

    # Create a multi-select widget for the teams
    selected_teams = st.multiselect('Select teams', team_stats['Team'].unique())

    # Filter data based on the selected teams
    filtered_data = team_stats[team_stats['Team'].isin(selected_teams)]

    fig = px.line(filtered_data, x='Year', y='Matches', color='Team',
                title="Number of Matches Played by Each Team Over the Years",
                labels={'Matches': 'Number of Matches', 'Year': 'Year'}, # renaming labels
                hover_data={"Year": True, "Matches": ':.2f'}) # hover data

    fig.update_layout(
        title_font_family="Arial",  # setting the title font
        title_font_color="RebeccaPurple",  # setting the title color
        title_font_size=24,  # setting the title size
        autosize=False,  # turn off autosize
        width=800,  # width
        height=500,  # height
        #paper_bgcolor="LightSteelBlue",  # setting the paper background color
    )

    fig.update_xaxes(
        title_text = 'Year',  # xaxis label
        tickangle = -45,  # xaxis label angle
        title_font = {"size": 14},  # xaxis label size
        title_standoff = 25)  # distance of the label from the axis

    fig.update_yaxes(
        title_text = 'Number of Matches',  # yaxis label
        title_font = {"size": 14},  # yaxis label size
        title_standoff = 25)  # distance of the label from the axis

    st.plotly_chart(fig)
