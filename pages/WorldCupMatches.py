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
    df_matches['Total Goals']      = df_matches['Home Team Goals'] + df_matches['Away Team Goals']

    df_matches = df_matches.dropna()

    #Clean the variables internal strings
    import re

    # This function will return the nationality found within brackets in a string.
    def extract_nationality_of_referees(referee):
        nationality = re.search(r'\((.*?)\)', referee)
        return nationality.group(1) if nationality else None # If no nationality is found, it will return None.

    # Create the new column 'Referee_nationality'
    df_matches['Referee_nationality'] = df_matches['Referee'].apply(extract_nationality_of_referees)

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

    #st.write(df_goals)
    
    st.markdown("### Check the summary statistics of the data")
    st.markdown("Select the checkboxes below to display the corresponding dataset:")
    
    #
    # --- DF_2) la partita con più goal della storia / la partita con più pubblico della storia --- #
    #

    # Create a DataFrame with the required information
    df_matches['Total Goals'] = df_matches['Home Team Goals'] + df_matches['Away Team Goals']
    max_goals_match = df_matches.loc[df_matches['Total Goals'].idxmax()]
    max_attendance_match = df_matches.loc[df_matches['Attendance'].idxmax()]
    combined_table_max_goals_attendance = pd.concat([max_goals_match, max_attendance_match], axis=1)
    combined_table_max_goals_attendance.columns = ["Match with the Most Goals in History", "Match with the Most Attendance in History"]

    #diplay in steamlit
    if st.checkbox("Display Match with the Most Goals and Attendance in WC History"):

        # Display the combined table
        st.header("Matches With Most Goals and Attendance in WC History")

        # Add table styling
        st.markdown("""
        <style>
        .dataframe {
            border: 2px solid black;
            background-color: #fafafa;
        }
        .dataframe tbody tr th {
            vertical-align: top;
            font-size: 16px;
            font-weight: bold;
        }
        .dataframe tbody tr td {
            font-size: 14px;
        }
        .dataframe thead th {
            text-align: center;
            background-color: #6c757d;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        st.table(combined_table_max_goals_attendance)
    
    #
    # --- DF_3) la squadra con più vittorie nella storia / la squadra con più sconfitte nella storia --- #
    #
            
    # Team with the most wins
    wins = pd.concat([df_matches[df_matches['Home Team Goals'] > df_matches['Away Team Goals']]['Home Team Name'],
                    df_matches[df_matches['Away Team Goals'] > df_matches['Home Team Goals']]['Away Team Name']])
    most_wins_team = wins.value_counts().idxmax()

    # Team with the most defeats
    defeats = pd.concat([df_matches[df_matches['Home Team Goals'] < df_matches['Away Team Goals']]['Home Team Name'],
                        df_matches[df_matches['Away Team Goals'] < df_matches['Home Team Goals']]['Away Team Name']])
    most_defeats_team = defeats.value_counts().idxmax()

    # Create a DataFrame with the required information
    combined_table_max_wins_losses = pd.DataFrame({
        "Record": ["Team with the Most Wins in History", "Team with the Most Defeats in History"],
        "Team": [most_wins_team, most_defeats_team]})

    # Set 'Record' as the index
    combined_table_max_wins_losses.set_index('Record', inplace=True)

    #display in steamlit
    if st.checkbox("Display Team with the Most Wins and Losses in WC History"):

        # Display the table with style
        st.header("Team With Most Wins and Losses in WC History")
        st.markdown("""
        <style>
        .dataframe {
            border: 2px solid black;
            background-color: #fafafa;
        }
        .dataframe tbody tr th {
            vertical-align: top;
            font-size: 16px;
            font-weight: bold;
        }
        .dataframe tbody tr td {
            font-size: 14px;
        }
        .dataframe thead th {
            text-align: center;
            background-color: #6c757d;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        st.table(combined_table_max_wins_losses)
   
    #
    # --- DF_4) la squadra con più vittorie nella storia / la squadra con più sconfitte nella storia 
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
    
    if st.checkbox("Display Summary Statistics by Team"):
        # Display the table in Streamlit
        st.header("Summary Statistics by Team: Matches Played, Wins, Losses, Win-Loss Ratio, Win-Match Ratio, Loss-Match Ratiovin WC History")
        st.table(team_stats_1)
   
    #
    # --- DF_5) la squadra con più vittorie nella storia / la squadra con più sconfitte nella storia - table by year --- #
    # 
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

    #display in steamlit
    if st.checkbox("Display Summary Statistics by Year"):
        st.table(team_stats)

    #
    # --- VS_1) linechart by year and country --- #
    #

    st.markdown("### Visualizations - Teams")
    
    st.markdown("Visual 1: Line chart of matches played by team over years")

    # Create a multi-select widget for the teams
    selected_teams_1 = st.multiselect('Select teams', team_stats['Team'].unique())

    # Filter data based on the selected teams
    filtered_data_1 = team_stats[team_stats['Team'].isin(selected_teams_1)]

    fig = px.line(filtered_data_1, x='Year', y='Matches', color='Team',
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

    #
    # --- VS_2) A heat map showing the number of wins for each team over the years --- #
    #

    st.markdown("Visual 2: Heatmap of wins by team over years")

    import plotly.graph_objects as go

    # Create a list of unique team names
    team_list_1 = team_stats['Team'].unique().tolist()

    # Team selection filter
    selected_teams_2 = st.multiselect('Select Teams', team_list_1)

    # Filter the team_stats DataFrame based on the selected teams
    filtered_stats_1 = team_stats[team_stats['Team'].isin(selected_teams_2)]

    # Pivot the filtered DataFrame
    team_stats_wide_1 = filtered_stats_1.pivot(index='Team', columns='Year', values='Wins').fillna(0)

    # Create the heatmap figure
    fig1 = go.Figure(data=go.Heatmap(z=team_stats_wide_1.values,
                                     x=team_stats_wide_1.columns,
                                     y=team_stats_wide_1.index,
                                     colorscale='RdYlGn'))

    # Set the title, x-label, y-label, and legend
    fig1.update_layout(
        title="Team Statistics",
        xaxis_title="Year",
        yaxis_title="Team",
        legend_title="Number of Wins"
    )

    # Label all years dynamically
    fig1.update_xaxes(type='category', tickangle=-45)

    # Display the figure using Plotly in Streamlit
    st.plotly_chart(fig1)


    #
    # --- VS_3) Pie chart Proportion of Wins and Losses to the Number of Matches for a Specific Team --- #
    #

    st.markdown("Visual 3: Proportion of Wins and Losses to the Number of Matches for a Specific Team")
    st.markdown("It is possible to select one team at time, each year or all year altogether, the staistics adapt")

    # Team selection filter
    team_select = st.selectbox('Select a team', team_stats['Team'].unique(), key="team_select_fig_2")

    # Add 'ALL' to year list
    years = list(team_stats['Year'].unique())
    years = ['ALL'] + years

    # Year selection filter
    year_select = st.selectbox('Select a year', years, key="year_select_fig_2")

    if year_select != 'ALL':
        selected_team_stats = team_stats[(team_stats['Team'] == team_select) & (team_stats['Year'] == year_select)]
    else:
        selected_team_stats = team_stats[team_stats['Team'] == team_select]

    #plot
    if len(selected_team_stats) == 0:
        st.error("No records found for the selected team and year.")
    else:
        # Calculate proportions
        total_matches = selected_team_stats['Matches'].sum()
        home_wins = selected_team_stats['Home Wins'].sum()
        away_wins = selected_team_stats['Away Wins'].sum()
        home_losses = selected_team_stats['Home Losses'].sum()
        away_losses = selected_team_stats['Away Losses'].sum()

        # Calculate proportions of wins and losses
        win_proportion = (home_wins + away_wins) / total_matches
        loss_proportion = (home_losses + away_losses) / total_matches

        # Create the pie chart figure
        fig2 = go.Figure(data=[go.Pie(labels=['Wins', 'Losses'], 
                                    values=[win_proportion, loss_proportion])])

        # Set the title
        fig2.update_layout(
            title=f"Proportion of Wins and Losses to the Number of Matches ({team_select} - {year_select})"
        )

        # Display the figure using Plotly in Streamlit
        st.plotly_chart(fig2)



    ### --- VS4 - REFEREES --- ###

    st.markdown("### Visualizations - Referees")
    st.markdown("It is possible to select one team at time, each year or all year altogether, the staistics adapt. You can either seect one nationality or a referee separately")
    
    import plotly.graph_objects as go

    # Add 'ALL' to year list
    years = list(df_matches['Year'].unique())
    years = ['ALL'] + years

    # Define selectboxes
    selected_year = st.selectbox('Select a year', years)

    # Filter the referees and nationalities based on the selected year
    if selected_year != 'ALL':
        referees_for_selected_year = ['EMPTY'] + list(df_matches[df_matches['Year'] == selected_year]['Referee'].unique())
        nationalities_for_selected_year = ['EMPTY'] + list(df_matches[df_matches['Year'] == selected_year]['Referee_nationality'].unique())
    else:
        referees_for_selected_year = ['EMPTY'] + list(df_matches['Referee'].unique())
        nationalities_for_selected_year = ['EMPTY'] + list(df_matches['Referee_nationality'].unique())

    selected_referee = st.selectbox('Select a referee', referees_for_selected_year)
    selected_nationality = st.selectbox('Select a nationality', nationalities_for_selected_year)

    # Filter data based on selected year
    if selected_year != 'ALL':
        filtered_year_matches = df_matches[df_matches['Year'] == selected_year]
    else:
        filtered_year_matches = df_matches

    # Filter data for referee and nationality
    if selected_referee != 'EMPTY':
        filtered_referee_matches = filtered_year_matches[filtered_year_matches['Referee'] == selected_referee]
        ref_group = selected_referee
    else:
        filtered_referee_matches = pd.DataFrame()

    if selected_nationality != 'EMPTY':
        filtered_nationality_matches = filtered_year_matches[filtered_year_matches['Referee_nationality'] == selected_nationality]
        nat_group = selected_nationality
    else:
        filtered_nationality_matches = pd.DataFrame()

    if not filtered_referee_matches.empty:
        # Plot 1: Number of matches refereed
        matches_refereed = filtered_year_matches['Referee'].value_counts()
        selected_referee_matches = matches_refereed[ref_group]
        fig1 = go.Figure(data=[go.Histogram(x=matches_refereed.values,
                                            marker_color='gray',
                                            name='All referees')])
        fig1.add_trace(go.Histogram(x=[selected_referee_matches],
                                    marker_color='red',
                                    name=ref_group))
        fig1.update_layout(barmode='overlay',
                        title_text='Number of matches refereed by referees',
                        xaxis_title='Number of matches',
                        yaxis_title='Number of referees')
        fig1.update_traces(opacity=0.75)
        st.plotly_chart(fig1)

    if not filtered_nationality_matches.empty:
        # Plot 2: Number of matches refereed
        matches_refereed = filtered_year_matches['Referee_nationality'].value_counts()
        selected_nationality_matches = matches_refereed[nat_group]
        fig2 = go.Figure(data=[go.Histogram(x=matches_refereed.values,
                                            marker_color='gray',
                                            name='All nationalities')])
        fig2.add_trace(go.Histogram(x=[selected_nationality_matches],
                                    marker_color='red',
                                    name=nat_group))
        fig2.update_layout(barmode='overlay',
                        title_text='Number of matches refereed by referees of selected nationality',
                        xaxis_title='Number of matches',
                        yaxis_title='Number of nationalities')
        fig2.update_traces(opacity=0.75)
        st.plotly_chart(fig2)
    else:
        st.write("No data to display. Please select a referee or a nationality.")








    '''
    #
    # --- VS_4) CREATE A RACING CHART --- #
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
    duration = 2000  # Slowing down the animation

    # Create the figure and axes objects
    fig, ax = plt.subplots(figsize=(15, 8))

    # Generate color map
    color_map = plt.get_cmap('tab20c')

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
            ax.set_xlim(0, df_year['Goals'].max() + 10)
            ax.set_ylim(-0.5, n_teams - 0.5)
            
            # Set the x and y ticks
            ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
            ax.yaxis.set_ticks(range(n_teams))
            ax.yaxis.set_ticklabels(df_year['Team'])
            
            # Set the title and labels
            ax.set_title(f'Total Goals Scored by Each Team: {year}', fontsize=24)
            ax.set_xlabel('Goals', fontsize=18)
            ax.set_ylabel('Team', fontsize=18)
            
            # Plot the bars with different colors
            bars = ax.barh(df_year['Team'], df_year['Goals'], color=color_map(np.linspace(0, 1, n_teams)))
            
            # Add value labels to the bars
            for bar in bars:
                width = bar.get_width()
                label = f'{width:.0f}'
                x = width + 1
                y = bar.get_y() + bar.get_height() / 2
                ax.text(x, y, label, ha='left', va='center', fontsize=14)
                
            # Add gridlines
            ax.grid(True, which='both', color='gray', linewidth=0.5)

    # Create the animation object
    start_year = int(df_goals['Year'].min())
    end_year = int(df_goals['Year'].max())
    animator = FuncAnimation(fig, draw_chart, frames=range(start_year, end_year + 1), interval= 5000)

    # Save animation as gif file 
    writer = PillowWriter(fps=200) 
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