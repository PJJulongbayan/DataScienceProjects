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

## LOAD DATA SET
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.csv'
df_can = pd.read_csv(url)
df_can.head()
print(df_can.shape)

# Set the country name as index - useful for quickly looking up countries using .loc method.
df_can.set_index('Country', inplace=True)

# Let's view the first five elements and see how the dataframe was changed
df_can.head()
print('data dimensions:', df_can.shape)

# finally, let's create a list of years from 1980 - 2013
# this will come in handy when we start plotting the data
years = list(map(str, range(1980, 2014)))
years

df_can.sort_values(['Total'], ascending=False, axis=0, inplace=True)
df_can


## LINE PLOT
# use code below to follow ggplot format instead
mpl.style.use(['ggplot'])

# transposed is used since index (country) is used by default. 
df_can.iloc[0:4,3:37].transpose().plot(kind='line')

## AREA PLOTS 

# get the top 5 entries
df_top5 = df_can.head()
# transpose the dataframe
df_top5 = df_top5[years].transpose()
df_top5.head()
# can use df.iat[rownum,colnum] for faster extraction of scalar values. 

# let's change the index values of df_top5 to type integer for plotting
df_top5.index = df_top5.index.map(int)
df_top5.plot(kind='area',
             stacked=False,
             figsize=(20, 10))  # pass a tuple (x, y) size

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')
plt.show()

# The unstacked plot has a default transparency (alpha value) at 0.5. 
# We can modify this value by passing in the alpha parameter.

df_top5.plot(kind='area', 
             alpha=0.25,  # 0 - 1, default value alpha = 0.5
             stacked=False,
             figsize=(20, 10))

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()

# Code above uses scripting layer (procedural method) - using matplotlib.pyplot as 'plt'
# Option 2: Artist layer (object oriented method) - using an axes instance from Matplotlib (preferred)

# option 2: preferred option with more flexibility
ax = df_top5.plot(kind='area', stacked=False, alpha=0.35, figsize=(20, 10))
ax.set_title('Immigration Trend of Top 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')


## HISTOGRAMS

# let's quickly view the 2013 data
df_can['2013'].head()

# we will us Numpy's histrogram method to get the bin ranges and frequency counts as follows:
# np.histogram returns 2 values (count and bin edges)
count, bin_edges = np.histogram(df_can['2013'])

print(count) # frequency count
print(bin_edges) # bin ranges, default = 10 bins

# By default, the histrogram method breaks up the dataset into 10 bins. The figure below summarizes
# the bin ranges and the frequency distribution of immigration in 2013. We can see that in 2013:

df_can['2013'].plot(kind='hist', figsize=(8, 5))
# add a title to the histogram
plt.title('Histogram of Immigration from 195 Countries in 2013')
# add y-label
plt.ylabel('Number of Countries')
# add x-label
plt.xlabel('Number of Immigrants')
plt.show()

# Notice that the x-axis labels do not match with the bin size.
# This can be fixed by passing in a xticks keyword that contains the list of the bin sizes, as follows:

# 'bin_edges' is a list of bin intervals
df_can['2013'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)

plt.title('Histogram of Immigration from 195 countries in 2013') # add a title to the histogram
plt.ylabel('Number of Countries') # add y-label
plt.xlabel('Number of Immigrants') # add x-label

plt.show()

# Side Note: We could use df_can['2013'].plot.hist(), instead. 
# Using some_data.plot(kind='type_plot', ...) is equivalent to some_data.plot.type_plot(...).
# That is, passing the type of the plot as argument or method behaves the same.

# Plotting the immigration distribution for Denmark, Norway, and Sweden for years 1980 - 2013.
# transpose dataframe
df_t = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
df_t.head()

# generate histogram
df_t.plot(kind='hist', figsize=(10, 6))

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()

# Let's make a few modifications to improve the impact and aesthetics of the previous plot:
    # increase the bin size to 15 by passing in bins parameter;
    # set transparency to 60% by passing in alpha parameter;
    # label the x-axis by passing in x-label parameter;
    # change the colors of the plots by passing in color parameter.

# let's get the x-tick values
count, bin_edges = np.histogram(df_t, 15)

# un-stacked histogram
df_t.plot(kind ='hist', 
          figsize=(10, 6),
          bins=15,
          alpha=0.6,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen']
         )

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')
plt.show()

# combined histogram may be misleading since it hides some graph information 
# If we do not want the plots to overlap each other, we can stack them using the stacked parameter. 
# Let's also adjust the min and max x-axis labels to remove the extra gap on the edges of the plot. 
# We can pass a tuple (min,max) using the xlim paramater, as show below.

count, bin_edges = np.histogram(df_t, 15)
xmin = bin_edges[0] - 10   #  first bin value is 31.0, adding buffer of 10 for aesthetic purposes 
xmax = bin_edges[-1] + 10  #  last bin value is 308.0, adding buffer of 10 for aesthetic purposes

# stacked Histogram
df_t.plot(kind='hist',
          figsize=(10, 6), 
          bins=15,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen'],
          stacked=True,
          xlim=(xmin, xmax)
         )

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants') 

plt.show()

## BAR CHARTS
# kind=bar creates a vertical bar plot
# kind=barh creates a horizontal bar plot

# The 2008 - 2011 Icelandic Financial Crisis was a major economic and political event in Iceland. 
# Relative to the size of its economy, Iceland's systemic banking collapse was the largest experienced 
# by any country in economic history. The crisis led to a severe economic depression in 2008 - 2011 and
# significant political unrest.

# step 1: get the data
df_iceland = df_can.loc['Iceland', years]
df_iceland.head()

# step 2: plot data
df_iceland.plot(kind='bar', figsize=(10, 6))

plt.xlabel('Year') # add to x-label to the plot
plt.ylabel('Number of immigrants') # add y-label to the plot
plt.title('Icelandic immigrants to Canada from 1980 to 2013') # add title to the plot

plt.show()

# Annotate iceland and restructure
df_iceland.plot(kind='bar', figsize=(10, 6), rot=90)  # rotate the xticks(labelled points on x-axis) by 90 degrees

plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.title('Icelandic Immigrants to Canada from 1980 to 2013')

# Annotate arrow
plt.annotate('',  # s: str. Will leave it blank for no text
             xy=(32, 70),  # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),  # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',  # will use the coordinate system of the object being annotated
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
             )

plt.show()


# Additional annotations
df_iceland.plot(kind='bar', figsize=(10, 6), rot=90)

plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.title('Icelandic Immigrants to Canada from 1980 to 2013')

# Annotate arrow
plt.annotate('',  # s: str. will leave it blank for no text
             xy=(32, 70),  # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),  # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',  # will use the coordinate system of the object being annotated
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
             )

# Annotate Text
plt.annotate('2008 - 2011 Financial Crisis',  # text to display
             xy=(28, 30),  # start the text at at point (year 2008 , pop 30)
             rotation=72.5,  # based on trial and error to match the arrow
             va='bottom',  # want the text to be vertically 'bottom' aligned
             ha='left',  # want the text to be horizontally 'left' algned.
             )

plt.show()

# Horizontal bar plot

# create a horizontal bar plot showing the total number of immigrants to Canada
# from the top 15 countries, for the period 1980 - 2013

# sort dataframe on 'Total' column (descending)
df_can.sort_values(by='Total', ascending=True, inplace=True)

# get top 15 countries
df_top15 = df_can['Total'].tail(15)
df_top15

# generate plot
df_top15.plot(kind='barh', figsize=(12, 12), color='steelblue')
plt.xlabel('Number of Immigrants')
plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')

# annotate value labels to each country
for index, value in enumerate(df_top15): 
    label = format(int(value), ',') # format int with commas

# place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
    plt.annotate(label, xy=(value - 47000, index - 0.10), color='white')

plt.show()

