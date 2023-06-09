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
import os
import branca.colormap as cm
from branca.element import MacroElement
from jinja2 import Template
# interactive histogram
import plotly.express as px
import plotly.graph_objects as go


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
    st.header("World Cups")

    # ---------------------- SIDEBAR ----------------------
    if st.sidebar.checkbox('Show Dataset'):
        st.subheader('World Cups')
        st.write(df_world_cups)

    print(df_world_cups)

    # ---------------------- 1_WORLD MAP ----------------------
    st.subheader("World Cup Performance by Country")
    # Load the dataset
    df_worldmap_cup = pd.DataFrame({
        'Country': ['Brazil', 'Italy', 'Germany FR', 'Uruguay', 'Argentina', 'England', 'France', 'Spain', 'Germany', 'Netherlands',
                    'Czechoslovakia', 'Hungary', 'Sweden', 'Poland', 'USA', 'Austria', 'Chile', 'Portugal', 'Croatia', 'Turkey',
                    'Yugoslavia', 'Soviet Union', 'Belgium', 'Bulgaria', 'Korea Republic'],
        'Winner': [5.0, 4.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'Second': [2.0, 2.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0, 3.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'Third': [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'Fourth': [2.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    })

    # Replace "Germany FR" with "Germany"
    df_worldmap_cup['Country'] = df_worldmap_cup['Country'].replace('Germany FR', 'Germany')
    # Group by "Country" and sum the values
    df_worldmap_cup = df_worldmap_cup.groupby('Country').sum().reset_index()

    # Load world shapefile data
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Set 'Country' column as the index
    df_worldmap_cup.set_index('Country', inplace=True)
    # Reshape the dataframe using melt function
    df_worldmap_cup = df_worldmap_cup.reset_index().melt(id_vars='Country', var_name='Placement', value_name='Count')
    # Merge the world shapefile data with the world cup data
    merged = world.merge(df_worldmap_cup, left_on='name', right_on='Country', how='left')
    # Fill NaN values in the 'Count' column with 0
    merged['Count'] = merged['Count'].fillna(0)
    # Define color scheme
    color_map = 'YlOrBr'
    # Create a folium map centered on the world
    world_map = folium.Map(tiles='cartodbpositron')

    # # Define the values and colors
    # values = [0, 1, 2, 3, 4, 5]
    # colors = ['#ffffff', '#CAF0F8', '#90E0EF', '#00B4D8', '#0077B6', '#03045E']

    # # Create a dictionary to map values to colors
    # value_color_map = dict(zip(values, colors))

    # # Create a custom function to map values to colors
    # def get_color(value):
    #     return value_color_map.get(value, 'white')

    # # Create a function to update the map based on the selected placement
    # def update_map(selected_placement):
    #     # Filter the merged dataframe based on the selected placement
    #     filtered_data = merged[merged['Placement'] == selected_placement]  
    #     # Create GeoJson layer for the filtered data
    #     geojson = folium.GeoJson(
    #         data=filtered_data,
    #         style_function=lambda feature: {
    #             'fillColor': get_color(feature['properties']['Count']),
    #             'fillOpacity': 0.7,
    #             'color': 'black',
    #             'weight': 1,
    #         }
    #     )
    #     # Add tooltips to the map
    #     tooltip = folium.GeoJsonTooltip(fields=['name', 'Count'], aliases=['Country:', 'Count:'])
    #     geojson.add_child(tooltip)
    #     # Add the GeoJson layer to the map
    #     geojson.add_to(world_map)

    # # Get the available placement options
    # placements = df_worldmap_cup['Placement'].unique()
    # # Create a dropdown menu to select the placement
    # selected_placement = st.selectbox('Select Placement:', placements)
    # # Update the map based on the selected placement
    # update_map(selected_placement)
    # # Display the map in Streamlit
    # st.components.v1.html(world_map._repr_html_(), height=500)

    # Define the values and colors
    values = [0, 1, 2, 3, 4, 5]
    colors = ['#ffffff', '#CAF0F8', '#90E0EF', '#00B4D8', '#0077B6', '#03045E']

    # Create a dictionary to map values to colors
    value_color_map = dict(zip(values, colors))

    # Create a custom function to map values to colors
    def get_color(value):
        return value_color_map.get(value, 'white')

    # Create a function to update the map based on the selected placement
    def update_map(selected_placement):
        # Filter the merged dataframe based on the selected placement
        filtered_data = merged[merged['Placement'] == selected_placement]  
        # Create GeoJson layer for the filtered data
        geojson = folium.GeoJson(
            data=filtered_data,
            style_function=lambda feature: {
                'fillColor': get_color(feature['properties']['Count']),
                'fillOpacity': 0.7,
                'color': 'black',
                'weight': 1,
            }
        )
        # Add tooltips to the map
        tooltip = folium.GeoJsonTooltip(fields=['name', 'Count'], aliases=['Country:', 'Count:'])
        geojson.add_child(tooltip)
        # Add the GeoJson layer to the map
        geojson.add_to(world_map)

    # Get the available placement options
    placements = df_worldmap_cup['Placement'].unique()
    # Create a dropdown menu to select the placement
    selected_placement = st.selectbox('Select Placement:', placements)
    # Update the map based on the selected placement
    update_map(selected_placement)

    # Create a template for the legend
    template = """
    {% macro html(this, kwargs) %}
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; font-size:14px; background-color: white; padding: 10px; border-radius: 3px;">
        <h5>Number of placements</h5>
        {% for value, color in this.value_color_map.items() %}
            <i style="background-color: {{color}}; width: 18px; height: 18px; float: left; margin-right: 8px;"></i> {{value}}<br>
        {% endfor %}
    </div>
    {% endmacro %}
    """

    # Create a custom legend element using the template and value-color map
    legend = MacroElement()
    legend._template = Template(template)
    legend.value_color_map = value_color_map

    # Add the legend element to the map
    world_map.get_root().add_child(legend)

    # Display the map in Streamlit
    st.components.v1.html(world_map._repr_html_(), height=500)



    # description
    st.markdown('''
                Explore the Rich Legacy of Football World Cup: An interactive world map showcasing 
                the historical success of national teams. Hover over countries to reveal their 
                achievements in the prestigious tournament, including triumphs as winners, runners-up, 
                third-place finishers, and fourth-place contenders. Discover the nations that have
                made their mark on the global stage of football.
                ''')

    # ---------------------- 2_NUMBER OF TIMES IN THE TOP 4 TEAMS ----------------------
    top_4 = ['Winner', 'Runners-Up', 'Third', 'Fourth']
    # merge "Germany FR" and "Germany" into "Germany"
    df_world_cups['Winner'] = df_world_cups['Winner'].replace('Germany FR', 'Germany')
    df_world_cups['Runners-Up'] = df_world_cups['Runners-Up'].replace('Germany FR', 'Germany')
    df_world_cups['Third'] = df_world_cups['Third'].replace('Germany FR', 'Germany')
    df_world_cups['Fourth'] = df_world_cups['Fourth'].replace('Germany FR', 'Germany')
    df_top_4 = df_world_cups.melt(id_vars='Year', value_vars=top_4, var_name='Position', value_name='Team')
    df_top_4_count = df_top_4.groupby('Team').count()[['Position']].reset_index().rename(columns={'Position': 'Count'})

    fig = px.bar(df_top_4_count, x='Team', y='Count', 
                 color='Team', 
                 title='Number of Times Each Team Finished in Top 4')

    fig.update_layout(
        xaxis=dict(
            tickangle=-45
        )
    )

    st.plotly_chart(fig)

    # ---------------------- 3_QUALIFIED TEAMS ----------------------
    st.subheader("Number of Qualified Teams by Year")
    # Group by year and country to get the count of qualified teams
    values = df_world_cups.groupby('Year')['QualifiedTeams'].sum().values

    # Define the custom colors
    custom_colors = ['#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF']

    # Create a color dictionary to assign colors based on values
    color_dict = {}
    color_index = 0

    for value in values:
        if value not in color_dict:
            color_dict[value] = custom_colors[color_index]
            color_index = (color_index + 1) % len(custom_colors)

    # Create the treemap
    fig = go.Figure(go.Treemap(
        labels=df_world_cups['Year'],
        parents=[""] * len(df_world_cups),
        values=values,
        branchvalues="total",
        marker=dict(
            colors=[color_dict[value] for value in values],
            showscale=False,
        ),
        textfont=dict(
            size=20,
        )
    ))

    # Customize the treemap layout
    fig.update_layout(
        title='Distribution of Qualified Teams by Year',
        width=600,
        height=600
    )

    st.plotly_chart(fig)

    st.markdown('''
                The chart shows the distribution of qualified teams by year, with each rectangular section
                representing a year from 1930 to 2014. The size of each section is proportional to the number 
                of qualified teams in that year. \n
                By hovering over each rectangle, you can see the number of qualified teams for that year.
                ''')

    # ---------------------- 4_HISTORY TREND ----------------------
    st.subheader("History Trend")
    # ------------- GOALS SCORED PER YEAR -------------
    fig = px.bar(df_world_cups,
        x="Year", 
        y="GoalsScored",
        title="Goals Scored per Year",
        labels={'x': 'Year', 'y': 'Number of Goals'}, 
    )
    # Set x-ticks every 4 years from 1930 to 2014
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(1930, 2015, 4),
            ticktext=np.arange(1930, 2015, 4),
            tickangle=-45
        )
    )

    # Set a different color for each bar
    fig.update_traces(marker_color=['#AA0DFE', '#3283FE', '#85660D', '#782AB6', '#565656', 
                                    '#1C8356', '#FA0087', '#F7E1A0', '#E2E2E2', '#1CBE4F', 
                                    '#C4451C', '#DEA0FD', '#FE00FA', '#325A9B', '#FEAF16', 
                                    '#F8A19F', '#90AD1C', '#F6222E', '#1CFFCE', '#2ED9FF', 
                                    '#B10DA1', '#C075A6', '#FC1CBF', '#B00068', '#FBE426', '#16FF32'])
    
    fig.update_layout(height=600)

    st.plotly_chart(fig)
    
    st.markdown('''
            This bar graph shows the trend in the number of goals scored per year from 1990 to 2014. 
            The data indicates an overall increasing trend, with the highest number of goals scored in 2014 
            and the lowest in 1990. \n
            By hovering over each bar, you can see the exact number of goals scored for that year.
            ''')


    # ------------- MATCHES PLAYED PER YEAR -------------
    # histogram of matches played per year
    fig = px.bar(
        x=df_world_cups["Year"], 
        y=df_world_cups["MatchesPlayed"],
        # color=df_world_cups["MatchesPlayed"],
        title="Matches Played per Year",
        labels={'x': 'Year', 'y': 'Number of Matches'},
        orientation='v',
        height=400
    )
    # Set x-ticks every 4 years from 1930 to 2014
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(1930, 2015, 4),
            ticktext=np.arange(1930, 2015, 4)
        )
    )

    # Set a different color for each bar
    fig.update_traces(marker_color=['#DEA0FD', '#FA0087', '#F6222E', '#1CFFCE', '#85660D', 
                                    '#FEAF16', '#2ED9FF', '#C075A6', '#B10DA1', '#1CBE4F',
                                    '#90AD1C', '#F7E1A0', '#FE00FA', '#B00068', '#325A9B', 
                                    '#782AB6', '#3283FE', '#C4451C', '#E2E2E2', '#1C8356', 
                                    '#565656', '#F8A19F', '#FC1CBF', '#B10DA1'])
    
    st.plotly_chart(fig)

    st.markdown('''
        This bar graph shows the number of matches played per year from 1930 to 2014. 
        The data indicates an overall increasing trend, with the highest number of matches played in 2014 and the lowest in 1930. \n
        By hovering over each bar, you can see the exact number of matches played for that year.
        ''')        

    # ------------- GOALS SCORED BY MATCHES PLAYED IN EACH YEAR -------------
    # histogram of goals scored by matches played in each year
    fig = px.bar(
        x=df_world_cups["Year"], 
        y=df_world_cups["GoalsScored"]/df_world_cups["MatchesPlayed"],
        color=df_world_cups["GoalsScored"]/df_world_cups["MatchesPlayed"],
        title="Goals Scored by Matches Played in each Year",
        labels={'x': 'Year', 'y': 'Goals per Match'},
        height=400)
    # Set x-ticks every 4 years from 1930 to 2014
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(1930, 2015, 4),
            ticktext=np.arange(1930, 2015, 4)
        )
    )

    # Set a different color for each bar
    fig.update_traces(marker_color=['#F6222E', '#DEA0FD', '#B00068', '#1C8356', '#1CFFCE', 
                                    '#782AB6', '#F7E1A0', '#1CBE4F', '#2ED9FF', '#FC1CBF', 
                                    '#90AD1C', '#FEAF16', '#C4451C', '#FE00FA', '#565656', 
                                    '#85660D', '#B10DA1', '#E2E2E2', '#F8A19F', '#3283FE', 
                                    '#AA0DFE', '#C075A6', '#325A9B', '#FA0087'])

    st.plotly_chart(fig)

    st.markdown('''
            This bar graph shows the trend in the number of goals scored per match in each year 
            from 1930 to 2014. The data indicates that there has been a slight decrease in 
            the number of goals scored per match over time. While there have been some variations, 
            the overall trend appears to be a gradual decline in the number of goals scored per match. \n
            By hovering over each bar, you can see the number of goals scored per match for that year.
                ''')

    # ---------------------- AVERAGE ATTENDANCE PER MATCH ----------------------
    df_world_cups['Attendance'] = df_world_cups['Attendance'].str.replace('.', '')
    df_world_cups['Attendance'] = pd.to_numeric(df_world_cups['Attendance'])
    df_world_cups['MatchesPlayed'] = pd.to_numeric(df_world_cups['MatchesPlayed'])
    df_world_cups['AvgAttendance'] = df_world_cups['Attendance'] / df_world_cups['MatchesPlayed']
    fig = px.line(df_world_cups, 
                  x='Year', 
                  y='AvgAttendance', 
                  title='Average Attendance per Match in Each World Cup')
    fig.update_xaxes(tickvals=df_world_cups['Year'], ticktext=df_world_cups['Year'])
    st.plotly_chart(fig)

    st.markdown('''
        This line graph shows the trend in the average attendance per match in each World Cup. 
        The data indicates that there has been a general increase in average attendance over time,
        with a peak at 1994. The line graph shows a clear upward trend. \n
        By hovering over the line, you can see the exact average attendance for that year.
        ''')

    # ---------------------- 5_HOSTING COUNTRIES ----------------------
    st.subheader("Hosting the World Cup")

    # ---------------------- PIE CHART ----------------------
    # Calculate the count of each hosting country
    country_counts = df_world_cups['Country'].value_counts()

    # Define the mapping of countries to continents
    country_to_continent = {
        'Italy': 'Europe',
        'France': 'Europe',
        'Brazil': 'South America',
        'Mexico': 'North America',
        'Germany': 'Europe',
        'Uruguay': 'South America',
        'Switzerland': 'Europe',
        'Sweden': 'Europe',
        'Chile': 'South America',
        'England': 'Europe',
        'Argentina': 'South America',
        'Spain': 'Europe',
        'USA': 'North America',
        'Korea/Japan': 'Asia',
        'South Africa': 'Africa'
    }

    # Calculate the count of each continent
    continent_counts = df_world_cups['Country'].map(country_to_continent).value_counts()

    # Create a selectbox to choose the plot: 
    plot_choice = st.selectbox('Do you want to show countries or continents?', ['Countries', 'Continents'])

    # Display the selected plot
    if plot_choice == 'Countries':
        # Create the pie chart for hosting countries
        fig = go.Figure(data=go.Pie(labels=country_counts.index, values=country_counts))
        fig.update_layout(title_text='Hosting Countries for the World Cup')
        st.plotly_chart(fig)
    else:
        # Create the pie chart for continents
        fig = go.Figure(data=go.Pie(labels=continent_counts.index, values=continent_counts))
        fig.update_layout(title_text='Hosting Continents for the World Cup')
        st.plotly_chart(fig)

    st.markdown('''
        This pie chart shows the distribution of World Cup hosting countries and continents.\n
        You can choose to show the hosting countries or the hosting continents by selecting
        the corresponding option in the dropdown menu. 
                ''')

    # BAR CHART
    # does the host country have an advantage?
    df_host_wins = df_world_cups[df_world_cups['Country'] == df_world_cups['Winner']]
    st.subheader("Does the host country have an advantage?")

    # show only the columns 'Year', 'Country', 'Winner'
    df_host_wins_to_show = df_host_wins[['Year', 'Country', 'Winner']]
    # st.write(df_host_wins_to_show)

    # count the times of the hosting country winning the World Cup
    df_host_wins_count = df_host_wins_to_show['Country'].value_counts()
    # count the times of the hosting country not winning the World Cup
    df_host_loses_count = country_counts - df_host_wins_count
    # create a dataframe to store the count of wins and loses
    df_host_wins_loses = pd.DataFrame({'Wins': df_host_wins_count, 'Loses': df_host_loses_count})
    # remove the rows with NaN values
    df_host_wins_loses = df_host_wins_loses.dropna()
    # create a bar chart to show the count of wins and loses
    fig = px.bar(df_host_wins_loses, x=df_host_wins_loses.index, y=['Wins', 'Loses'], 
                title='Wins and Loses of Hosting Countries',
                # color of wins and loses
                color_discrete_map={'Wins': 'green', 'Loses': 'red'}
                )
    fig.update_xaxes(title='Hosting Countries')
    fig.update_yaxes(title='World Cup Won', tickformat='linear', dtick=1)

    st.plotly_chart(fig)

    st.markdown('''
        This bar graph shows the number of World Cup wins and losses for countries that have hosted the World Cup. 
        The data indicates that the host country does have an advantage, as all the countries shown have more wins than losses. 
        ''')

    # -------------------------- 5_STADIUM --------------------------
    
    # Calculate the mean attendance for each stadium and city
    stadium = df_matches[['Stadium', 'City', 'Attendance']].groupby(['Stadium', 'City']).mean().reset_index()

    # replace 'Maracan� - Est�dio Jornalista M�rio Filho' with 'Maracanã - Estádio Jornalista Mário Filho'
    stadium['Stadium'] = stadium['Stadium'].replace('Maracan� - Est�dio Jornalista M�rio Filho', 'Maracanã - Estádio Jornalista Mário Filho')

    # Sort the stadiums by attendance in descending order
    stadium = stadium.sort_values('Attendance', ascending=False).round(0)

    # Display the stadium DataFrame
    # st.write(stadium)

    # Get the top 10 stadiums with the highest attendance
    top_10_stadiums = stadium.head(10)

    # Reverse the order of the DataFrame so that the bar chart is ordered from top to bottom
    top_10_stadiums = top_10_stadiums.iloc[::-1]

    # Plot the top 10 stadiums in a horizontal bar chart
    fig = go.Figure(go.Bar(
        x=top_10_stadiums['Attendance'],
        y=top_10_stadiums['Stadium'],
        orientation='h',
        marker=dict(
            color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
        ),
        text=top_10_stadiums['Attendance'],
        hovertext = top_10_stadiums['City'],
        hovertemplate = 'Stadium: %{y} <br>City: %{hovertext} <br>Attendance: %{x} <extra></extra>'
    ))

    fig.update_layout(
        title='Top 10 Stadiums with Highest Attendance',
        xaxis=dict(
            title='Attendance'
        ),
        yaxis=dict(
            title='Stadium'
        ),
        width=800,
        height=600
    )

    st.plotly_chart(fig)

    st.markdown('''
        This bar graph shows the top 10 stadiums with the highest attendance. \n
        By hovering over each bar, you can see the city where the stadium is located.
        ''')

    