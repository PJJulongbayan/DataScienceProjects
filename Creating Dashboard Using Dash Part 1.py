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
import dash
from dash import dcc
from dash import html

# pip install packaging
# pip install pandas dash
#pip install dash plotly

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

# Pie Chart Creation
fig = px.pie(data, values='Flights', names='DistanceGroup', title='Distance group proportion by flights')
fig.show()


## CREATING DASH APPLICATION
# Create a dash application
app = dash.Dash(__name__)

# Get the layout of the application and adjust it.
# Create an outer division using html.Div and add title to the dashboard using html.H1 component
# Add description about the graph using HTML P (paragraph) component
# Finally, add graph component.

app.layout = html.Div(children=[html.H1('Airline Dashboard',style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
                                html.P('Proportion of distance group (250 mile distance interval group) by flights.', style={'textAlign':'center', 'color': '#F57241'}),
                                dcc.Graph(figure=fig),                                           
                    ])

# Run the application                   
if __name__ == '__main__':
    app.run_server()
