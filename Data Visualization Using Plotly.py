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
import plotly.express as px
import plotly.graph_objects as go

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
 
## LOAD DATASET

# The Reporting Carrier On-Time Performance Dataset contains information on 
# approximately 200 million domestic US flights reported to the United States
# Bureau of Transportation Statistics. The dataset contains basic information 
# about each flight (such as date, time, departure airport, arrival airport) and, 
# if applicable, the amount of time the flight was delayed and information about 
# the reason for the delay.
# This dataset can be used to predict the likelihood of a flight arriving on time.

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv'
airline_data = pd.read_csv(url,
                        encoding = "ISO-8859-1",
                        dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                'Div2Airport': str, 'Div2TailNum': str})

# Randomly sample 500 data points. Setting the random state to be 42 for reproducibility
data = airline_data.sample(n=500, random_state=42)

data.head()
data.shape

## Visualization 

# Scatter Plot (using plotly.graph_object)
# Create a scatter plot to represent departure time changes with respect to airport distance.

# First we will create an empty figure using go.Figure()
fig=go.Figure()

# Next we will create a scatter plot by using the add_trace function and use the go.scatter() function within it
# In go.Scatter we define the x-axis data,y-axis data and define the mode as markers with color of the marker as red

fig.add_trace(go.Scatter(x=data['Distance'], y=data['DepTime'], mode='markers', marker=dict(color='red')))
fig.update_layout(title='Distance vs Departure Time', xaxis_title='Distance', yaxis_title='DepTime')
fig.show()

# Line Plot (using plotly.graph_object)
# Using a line plot to extract average monthly arrival delay time
# and see how it changes over the year.

# Group the data by Month and compute average over arrival delay time.
line_data = data.groupby('Month')['ArrDelay'].mean().reset_index()
# Display the data
line_data

fig=go.Figure()
fig.add_trace(go.Scatter(x=line_data['Month'], y=line_data['ArrDelay'], mode='lines', marker=dict(color='green')))
fig.update_layout(title='Month vs Average Flight Delay Time', xaxis_title='Month', yaxis_title='ArrDelay')
fig.show()

# Month of June has the maximum average monthly delay time


# Bar Chart (using plotly.express)
# Using a bar chart to extract number of flights from a specific airline that goes to a destination
# Group the data by destination state and reporting airline. Compute total number of flights in each combination
bar_data = data.groupby('DestState')['Flights'].sum().reset_index()

fig = px.bar(bar_data, x="DestState", y="Flights", title='Total number of flights to the destination state split by reporting airline') 
fig.show()

# Maximum flights are to destination state CA which is around 
# 68 and there is only 1 flight to destination state VT


# Histogram (using plotly.express)

# Represent the distribution of arrival delay using a histogram
# Set missing values to 0
data['ArrDelay'] = data['ArrDelay'].fillna(0)
fig = px.histogram(data, x="ArrDelay",title="Total number of flights to the destination state split by reporting air.")
fig.show()

# only max of 5 flights with an arrival delay of 50-54 minutes and
# around 17 flights with an arrival delay of 20-25 minutes 

# Bubble Chart (using plotly.express)

# Using a bubble plot to represent number of flights as per reporting airline
bub_data = data.groupby('Reporting_Airline')['Flights'].sum().reset_index()
fig = px.scatter(bub_data, x="Reporting_Airline", y="Flights", size="Flights",
                 hover_name="Reporting_Airline", title='Reporting Airline vs Number of Flights', size_max=60)
fig.show()

# It is found that the reporting airline WN
# has the highest number of flights which is around 86

# Pie Chart (using plotly.express)

# Representing the proportion of Flights by Distance Group (Flights indicated by numbers)
fig = px.pie(data, values='Flights', names='DistanceGroup', title='Flight propotion by Distance Group')
fig.show()

# Distance group 2 has the highest flight proportion.

# SunBurst Charts (using plotly.express)
# Represent the hierarchical view in othe order of month and 
# destination state holding value of number of flights

fig = px.sunburst(data, path=['Month', 'DestStateName'], values='Flights',title='Flight Distribution Hierarchy')
fig.show()

#  Month numbers present in the innermost concentric circle is the root and for 
# each month we will check the number of flights for the different destination states under it.