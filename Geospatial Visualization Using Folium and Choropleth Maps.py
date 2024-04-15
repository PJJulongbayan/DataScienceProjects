# import standard libraries

import pandas as pd
import numpy as np
import csv
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats
import warnings
import matplotlib.patches as mpatches # needed for waffle Charts
from PIL import Image # converting images into arrays
from wordcloud import WordCloud, STOPWORDS
from pywaffle import Waffle
import folium
from folium import Choropleth

# optional: for ggplot-like style
mpl.style.use('ggplot') 

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# plt is the scrpting interface

# backend command below
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
# %matplotlib inline

# show all in pandas view
pd.set_option('display.max_columns', None)

## definining the world map
world_map = folium.Map()
world_map

# define the world map centered around Canada with a low zoom level
# higher zoom level provides more detail (focused more on the center)
world_map = folium.Map(location=[56.130, -106.35], zoom_start=4)
world_map

# create a Cartodb dark_matter map of the world centered around Canada
world_map = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Cartodb dark_matter')
world_map

# create a Cartodb positron map of the world centered around Canada
world_map = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Cartodb positron')
world_map

## Maps with Markers

# download dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Police_Department_Incidents_-_Previous_Year__2016_.csv"
df_incidents = pd.read_csv(url)
df_incidents.head()

df_incidents.shape

# reducing dataset from 150k rows
# get the first 100 crimes in the df_incidents dataframe
limit = 100
df_incidents = df_incidents.iloc[0:limit, :]

# Let's visualize where these crimes took place in the city of San Francisco. We will use the default style, 
# and we will initialize the zoom level to 12.
# San Francisco latitude and longitude values
latitude = 37.77
longitude = -122.42

# create map and display it
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)
sanfran_map

# Superimposing the locations of the crimes onto the map. The way to do that in Folium is to create 
# a feature group with its own features and style and then add it to the sanfran_map.

# instantiate a feature group for the incidents in the dataframe
incidents = folium.map.FeatureGroup()

# loop through the 100 crimes and add each to the incidents feature group
for lat, lng, in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        folium.vector_layers.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )

# add incidents to map
sanfran_map.add_child(incidents)

# instantiate a feature group for the incidents in the dataframe
incidents = folium.map.FeatureGroup()

# loop through the 100 crimes and add each to the incidents feature group
for lat, lng, in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        folium.vector_layers.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )

# add incidents to map
sanfran_map.add_child(incidents)
sanfran_map

# add pop-up text to each marker on the map
latitudes = list(df_incidents.Y)
longitudes = list(df_incidents.X)
labels = list(df_incidents.Category)

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.Marker([lat, lng], popup=label).add_to(sanfran_map)    

# add incidents to map
sanfran_map.add_child(incidents)

# Fixing map to avoid congestion

# create map and display it
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)
# loop through the 100 crimes and add each to the map
for lat, lng, label in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
    folium.vector_layers.CircleMarker(
        [lat, lng],
        radius=5, # define how big you want the circle markers to be
        color='yellow',
        fill=True,
        popup=label,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(sanfran_map)

# show map
sanfran_map

## Using marker cluster
from folium import plugins

# San Francisco latitude and longitude values
latitude = 37.77
longitude = -122.42

# let's start again with a clean copy of the map of San Francisco

sanfran_map_new = folium.Map(location = [latitude, longitude], zoom_start = 12)

# instantiate a mark cluster object for the incidents in the dataframe
incidents_new = plugins.MarkerCluster().add_to(sanfran_map_new)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents_new)

# display map
sanfran_map_new

## CHLOREPLETH MAPS
# read data

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv'
df_can = pd.read_csv(url)

# Need GeoJson file to use chorepleth maps

# download countries geojson file
world_geo = r'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/world_countries.json' # geojson file

# create a plain world map
world_map = folium.Map(location=[0, 0], zoom_start=2)

# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013
folium.Choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada', 
    reset=True).add_to(world_map)

# display map
world_map

































