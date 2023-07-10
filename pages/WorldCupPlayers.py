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
import plotly.colors


def run():
    # import datasets
    df_matches = pd.read_csv("WorldCupMatches.csv")
    df_players = pd.read_csv("WorldCupPlayers.csv")
    df_world_cups = pd.read_csv("WorldCups.csv")

    st.title("World Cup Data Visualization")
    st.header("World Cups Players")

    # data cleaning
    df_merge = df_matches[['Year', "MatchID"]]
    merged_df = pd.merge(df_players, df_merge, on='MatchID', how='left')
    df_players = merged_df
    df_players['Player Name'] = df_players['Player Name'].str.normalize('NFKD')
    df_players.drop_duplicates(inplace=True)

    # split the event column
    df_players['Event'].fillna('', inplace=True)
    df_players['Goals'] = df_players['Event'].apply(
        lambda x: sum(1 for event in x.split() if event.startswith('G')))
    df_players['OwnGoals'] = df_players['Event'].apply(
        lambda x: sum(1 for event in x.split() if event.startswith('OG')))
    df_players['Penalties'] = df_players['Event'].apply(
        lambda x: sum(1 for event in x.split() if event.startswith('P')))
    df_players['PenaltiesMissed'] = df_players['Event'].apply(
        lambda x: sum(1 for event in x.split() if event.startswith('MP')))
    df_players['YellowCards'] = df_players['Event'].apply(
        lambda x: sum(1 for event in x.split() if event.startswith('Y')))
    df_players['RedCards'] = df_players['Event'].apply(
        lambda x: sum(1 for event in x.split() if event.startswith('R')))
    df_players['RedCardsByYellow'] = df_players['Event'].apply(
        lambda x: sum(1 for event in x.split() if event.startswith('SY')))
    df_players['SubstituteIn'] = df_players['Event'].apply(
        lambda x: sum(1 for event in x.split() if event.startswith('I')))
    df_players['SubstituteOut'] = df_players['Event'].apply(
        lambda x: sum(1 for event in x.split() if event.startswith('O')))
    df_players['Minute'] = df_players['Event'].str.extract(
        '(\d+)').astype(float).fillna(0).astype(int)
    df_players['Team Initials'] = df_players['Team Initials'].replace(
        'FRG', 'GER')
    df_players['Team Initials'] = df_players['Team Initials'].replace(
        'URS', 'RUS')
    df_players['Team Initials'] = df_players['Team Initials'].replace(
        'TCH', 'CZE')
    # _______________________________________________________________________________________
    st.subheader("Number of Players by Team")
    # Barplot number of players by team
    team_counts = df_players.groupby('Team Initials')[
        'Player Name'].count().reset_index()

    # Sorting teams based on the number of players
    team_counts = team_counts.sort_values('Player Name', ascending=False)

    # Creating the bar chart using Plotly Express
    fig = px.bar(team_counts, x='Team Initials', y='Player Name',
                 title='Number of Players by Team', color='Team Initials')

    # Updating hover template to show 'Number of Players'
    fig.update_traces(hovertemplate='Number of Players: %{y}')
    fig.update_layout(xaxis=dict(tickangle=45, tickfont=dict(size=10)))
    fig.update_layout(bargap=0.2)
    st.write(fig)
    # _______________________________________________________________________________________
    st.subheader("Goals")
    # Barplot number of goals by team
    team_goals = df_players.groupby('Team Initials')[
        'Goals'].sum().reset_index()

    # Filtering out teams with 0 goals
    team_goals = team_goals[team_goals['Goals'] > 10]

    # Sorting teams based on total goals in descending order
    team_goals = team_goals.sort_values('Goals', ascending=False)

    # Creating the bar chart using Plotly Express
    fig1 = px.bar(team_goals, x='Team Initials', y='Goals',
                  title='Total Goals by Team', color="Team Initials")
    fig1.update_layout(xaxis=dict(tickangle=45, tickfont=dict(size=10)))
    fig1.update_layout(bargap=0.2)

    st.write(fig1)
    # _______________________________________________________________________________________

    # Barplot of top 3 goal scorers per team
    player_goals = df_players.groupby(['Team Initials', 'Player Name'])[
        'Goals'].sum().reset_index()

    # Selecting the top three goal scorers for each team
    top_scorers = player_goals.groupby('Team Initials').apply(
        lambda x: x.nlargest(3, 'Goals')).reset_index(drop=True)

    # Grouping the top scorers by team and aggregating the goals
    team_scorers = top_scorers.groupby('Team Initials').apply(
        lambda x: {'Player Name': x['Player Name'].tolist(), 'Goals': x['Goals'].tolist()}).to_dict()

    # Sorting teams based on total goals
    sorted_teams = sorted(team_scorers.keys(), key=lambda x: sum(
        team_scorers[x]['Goals']), reverse=True)

    # Limiting the number of teams to display
    num_teams = 30  # Choose the desired number of teams to display
    limited_teams = sorted_teams[:num_teams]

    # Creating the bar chart
    data = []
    for team in limited_teams:
        scorers = team_scorers[team]
        hover_text = []
        for name, goals in zip(scorers['Player Name'], scorers['Goals']):
            hover_text.append(f"Player Name: {name}<br>Goals: {goals}")
        data.append(go.Bar(x=[team]*3, y=scorers['Goals'], name=team,
                    hovertext=hover_text, textposition="none", width=0.5,
                    marker=dict(line=dict(color='white', width=1))))

    layout = go.Layout(barmode='group', title='Top 3 Goal Scorers per Team', xaxis={
                       'title': 'Team'}, yaxis={'title': 'Goals'})
    fig = go.Figure(data=data, layout=layout)

    # Configure hover mode and tooltip behavior
    fig.update_traces(hovertemplate='%{hovertext}')
    fig.update_layout(xaxis_tickangle=45)
    st.write(fig)

    # _______________________________________________________________________________________

    # Top goalscorer by year
    player_goals = df_players.groupby(['Player Name', 'Year', 'Team Initials'])[
        'Goals'].sum().reset_index()

    # Sort the data by year and goals in descending order
    player_goals = player_goals.sort_values(
        by=['Year', 'Goals'], ascending=[True, False])

    # Select the top goalscorer for each year
    top_scorers = player_goals.groupby('Year').first().reset_index()

    max_year = top_scorers['Year'].max()

    # Define the tick values at every 4-year interval
    tick_values = list(range(1930, int(max_year)+1, 4))

    # Create the scatter plot
    fig = px.scatter(top_scorers, x='Year', y='Goals', color='Team Initials',
                     title='Top Goalscorer by Year', labels={'Team Initials': 'Team'},
                     custom_data=['Player Name'])

    # Customize marker settings for better visibility
    fig.update_traces(mode='markers', marker=dict(size=8, symbol='circle'),
                      hovertemplate='Player: %{customdata[0]}<br>Goals: %{y}')

    # Set the x-axis tick values and labels
    fig.update_xaxes(tickmode='array', tickvals=tick_values,
                     ticktext=[str(year) for year in tick_values])

    fig.update_layout(xaxis_tickangle=45)
    st.write(fig)

    # _______________________________________________________________________________________

    # goals by shirt number
    goalscorers = df_players.groupby('Shirt Number')[
        'Goals'].sum().reset_index()

    # Sort the data in descending order based on the goals scored
    goalscorers = goalscorers.sort_values('Goals', ascending=False)
    goalscorers = goalscorers[goalscorers['Shirt Number'] != 0]

    # Create the bar chart
    fig = px.bar(goalscorers, y='Shirt Number', x='Goals',
                 orientation='h', title='Top Goalscorers by Shirt Number')

    # Set the axis labels
    fig.update_xaxes(title='Goals')
    fig.update_yaxes(title='Shirt Number')
    fig.update_layout(height=600)

    # Show all the shirt numbers on the y-axis
    fig.update_layout(yaxis=dict(tickmode='linear', dtick=1))

    # Show the graph
    st.write(fig)

    substitute_goals = df_players.loc[df_players['SubstituteIn'] > 0, 'Goals'].sum(
    )
    non_substitute_goals = df_players.loc[df_players['SubstituteIn'] == 0, 'Goals'].sum(
    )

    # Create the pie chart
    fig = go.Figure(data=[go.Pie(labels=['Substitutes', 'Non-Substitutes'],
                                 values=[substitute_goals,
                                         non_substitute_goals],
                                 hovertemplate='Goals: %{value}',
                                 marker=dict(colors=['red', 'blue']))])

    fig.update_traces(marker=dict(colors=['red']))
    # Set the chart title
    fig.update_layout(
        title='Percentage of Goals Scored by Substitutes and Non-Substitutes')

    st.write(fig)
    # _______________________________________________________________________________________

    # _______________________________________________________________________________________
    st.subheader("Penalties")
    # penalties
    penalties_data = df_players.groupby('Team Initials').agg(
        {'PenaltiesMissed': 'sum', 'Penalties': 'sum'}).reset_index()

    # Filter out teams with no penalties
    penalties_data = penalties_data[(penalties_data['PenaltiesMissed'] > 0) | (
        penalties_data['Penalties'] > 0)]

    # Calculate the total penalties for each team
    penalties_data['Total Penalties'] = penalties_data['Penalties'] + \
        penalties_data['PenaltiesMissed']

    # Sort the data in descending order based on the total penalties
    penalties_data = penalties_data.sort_values(
        'Total Penalties', ascending=False)

    # Create the bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=penalties_data['PenaltiesMissed'],
        y=penalties_data['Team Initials'],
        name='Missed Penalties',
        orientation='h',
        marker=dict(color='red')
    ))
    fig.add_trace(go.Bar(
        x=penalties_data['Total Penalties'],
        y=penalties_data['Team Initials'],
        name='Total Penalties',
        orientation='h',
        marker=dict(color='blue')
    ))

    fig.update_layout(yaxis_autorange='reversed')

    # Set the axis labels and title
    fig.update_layout(
        xaxis_title='Count',
        yaxis_title='Team',
        title='Penalties and Missed Penalties by Team'
    )

    # Adjust the height of the graph
    fig.update_layout(height=1000)

    st.write(fig)
    # _______________________________________________________________________________________

    st.subheader("Cards")
    player_cards = df_players.groupby('Player Name').agg(
    {'YellowCards': 'sum', 'RedCards': 'sum', 'Team Initials': 'first'}).reset_index()

    # Sort the players by the total number of cards
    player_cards.sort_values(
        by=['YellowCards', 'RedCards'], ascending=False, inplace=True)

    # Select the top 20 players
    top_players = player_cards.head(20)

    # Calculate the total number of cards for each player
    top_players['TotalCards'] = top_players['YellowCards'] + top_players['RedCards']

    # Assign a random color to each team
    color_palette = plotly.colors.qualitative.Light24

    # Create the color mapping dictionary for teams and colors
    team_colors = {team: color_palette[i % len(color_palette)] for i, team in enumerate(
        top_players['Team Initials'].unique())}

    # Create the TreeMap
    fig = go.Figure(go.Treemap(
        labels=top_players['Team Initials'] +
        ' - ' + top_players['Player Name'],
        parents=[''] * len(top_players),
        values=top_players['YellowCards'] + 1.5 * top_players['RedCards'],
        customdata=top_players[['YellowCards', 'RedCards']],
        hovertemplate='Player: %{label}<br>'
                  'Yellow Cards: %{customdata[0]}<br>'
                  'Red Cards: %{customdata[1]}',
        marker=dict(
            colors=[team_colors[team] for team in top_players['Team Initials']]),
        level=[2] * len(top_players)
    ))

    # Set the layout and title
    fig.update_layout(
        title='Top 20 Players with the Most Yellow and Red Cards',
    )
    fig.update_layout(height=500)

    st.write(fig)







    # _______________________________________________________________________________________

    # Scatterplot of goals vs cards
    # Group the data by team initials and calculate the sum of goals, red cards, and yellow cards
    team_data = df_players.groupby('Team Initials').agg(
        {'Goals': 'sum', 'RedCards': 'sum', 'YellowCards': 'sum'}).reset_index()

    # Fit linear regression for yellow cards
    yellow_reg = LinearRegression()
    yellow_reg.fit(team_data['Goals'].values.reshape(-1, 1),
                   team_data['YellowCards'].values.reshape(-1, 1))
    yellow_line = yellow_reg.predict(
        np.array([[0], [team_data['Goals'].max()]]))

    # Fit linear regression for red cards
    red_reg = LinearRegression()
    red_reg.fit(team_data['Goals'].values.reshape(-1, 1),
                team_data['RedCards'].values.reshape(-1, 1))
    red_line = red_reg.predict(np.array([[0], [team_data['Goals'].max()]]))

    # Create the scatter plot
    fig = go.Figure()

    # Add scatter plot for yellow cards
    fig.add_trace(go.Scatter(
        x=team_data['Goals'],
        y=team_data['YellowCards'],
        mode='markers',
        name='Yellow Cards',
        marker=dict(color='yellow'),
        hovertemplate='<b>Team:</b> %{text}<br>'
        '<b>Goals:</b> %{x}<br>'
        '<b>Yellow Cards:</b> %{y}<br>',
        text=team_data['Team Initials']
    ))

    # Add scatter plot for red cards
    fig.add_trace(go.Scatter(
        x=team_data['Goals'],
        y=team_data['RedCards'],
        mode='markers',
        name='Red Cards',
        marker=dict(color='red'),
        hovertemplate='<b>Team:</b> %{text}<br>'
        '<b>Goals:</b> %{x}<br>'
        '<b>Red Cards:</b> %{y}<br>',
        text=team_data['Team Initials']
    ))

    # Add the trend lines
    fig.add_trace(go.Scatter(
        x=[0, team_data['Goals'].max()],
        y=yellow_line.flatten(),
        mode='lines',
        name='Yellow Cards Trend',
        line=dict(color='yellow')
    ))

    fig.add_trace(go.Scatter(
        x=[0, team_data['Goals'].max()],
        y=red_line.flatten(),
        mode='lines',
        name='Red Cards Trend',
        line=dict(color='red')
    ))

    # Set the axis labels and title
    fig.update_layout(
        xaxis_title='Goals Scored',
        yaxis_title='Number of Cards',
        title='Relationship between Goals Scored and Red/Yellow Cards'
    )

    st.write(fig)
    # _______________________________________________________________________________________

    # bubbleplot cards vs goals
    team_data = df_players.groupby('Team Initials').agg(
        {'Goals': 'sum', 'YellowCards': 'sum', 'RedCards': 'sum'}).reset_index()

    # Create the bubble chart
    team_players = df_players.groupby('Team Initials')[
        'Player Name'].count().reset_index()
    team_players.rename(columns={'Player Name': 'NumPlayers'}, inplace=True)

    # Merge the player count with the team data
    team_data = pd.merge(team_data, team_players, on='Team Initials')

    # Create the bubble chart
    fig = px.scatter(team_data, x='YellowCards', y='RedCards', size='Goals', color='NumPlayers',
                     hover_name='Team Initials', log_x=True, log_y=True,
                     title='Relationship between Goals, Yellow Cards, Red Cards, and Number of Players',
                     color_continuous_scale='viridis')

    # Set the axis labels
    fig.update_xaxes(title='Yellow Cards')
    fig.update_yaxes(title='Red Cards')

    # Customize marker size
    fig.update_traces(marker=dict(sizemode='diameter', sizeref=2, sizemin=3))
    st.write(fig)
    # _______________________________________________________________________________________
