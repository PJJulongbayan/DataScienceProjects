# import standard libraries

import pandas as pd
import numpy as np
import csv
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats
import warnings

# Suppress matplotlib user warnings
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# plt is the scrpting interface

# backend command below
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
# %matplotlib inline

# show all in pandas view
pd.set_option('display.max_columns', None)

# Generate 1000 random numbers using pyplot
plt.hist(np.random.randn(1000))
plt.ylabel("Frequency")
plt.xlabel("Values")
plt.title("Plotting Example")
plt.show()

## LOAD DATA SET
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx'
df_can = pd.read_excel(url, 
    sheet_name='Canada by Citizenship',
    skiprows=range(20),
    skipfooter=2)

df_can.head()

# view rows and column index
df_can.columns.tolist()
df_can.index.tolist()
df_can.shape

## CLEANING DATA

# removing unnecessary columns
# in pandas axis=0 represents rows (default) and axis=1 represents columns.
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)
df_can.head(2)

# Let's rename the columns so that they make sense. We can use rename() method 
# by passing in a dictionary of old and new names as follows:
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
df_can.columns

df_can.head()

# sum the numeric columns and create a total
df_can['Total'] = df_can.iloc[:,4:43].sum(axis = 1)

# check if there are nulls
df_can.isnull().sum()

# describe dataset, excludes categorical
df_can.describe()

## INDEXING DATA

# df.loc[label]    # filters by the labels of the index/column
# df.iloc[index]   # filters by the positions of the index/column

# Setting the 'Country' column as the index using set_index() method.
# tip: The opposite of set is reset. So to reset the index, we can use df_can.reset_index()
df_can.set_index('Country', inplace=True)
df_can

# optional: to remove the name of the index
# df_can.index.name = None

# number of immigrants from Japan
df_can.loc['Japan']

# alternate methods
df_can.iloc[87]

# select Japan data as df
df_can[df_can.index == 'Japan']

# for year 2013
df_can.loc['Japan', 2013]

#  for years 1980 to 1985
df_can.loc['Japan', [1980, 1981, 1982, 1983, 1984, 1985]]

# Indexing for Haiti
df_can[df_can.index == 'Haiti'] # as df
df_can.loc['Haiti'] # as series

df_can.columns

# print type of data for each column
# for column in df_can.columns:
    # print(f"Column '{column}' has data type: {df_can[column].dtype}")

# change data type of each column in df
# for column in df_can.columns:
    # df_can[column] = df_can[column].astype('str')

## SORTING

# Let's sort out dataframe df_can on 'Total' column, in descending order 
# to find out the top 5 countries that contributed the most to immigration to Canada.

df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True, ignore_index = False)
top_5 = df_can.head(5)
top_5

## VISUALIZATION

# ggplot like-style of plotting 
print(plt.style.available)
mpl.style.use(['ggplot'])

# let's declare a variable that will 
# allow us to easily call upon the full range of years for use using .loc instead of iloc
years = list(map(str, range(1980, 2014)))
years

haiti = df_can.iloc[df_can.index.get_loc('Haiti'), 3:37]
haiti

# plotting 

haiti.plot(kind='line')
plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')
plt.show

# We can clearly notice how number of immigrants from Haiti spiked up from 
# 2010 as Canada stepped up its efforts to accept refugees from Haiti. 
# Let's annotate this spike in the plot by using the plt.text() method.

# annotate the 2010 Earthquake. 
# syntax: plt.text(x, y, label)
haiti.plot(kind='line')
plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')

plt.text(20, 6000, '2010 Earthquake') # see note below
plt.show

