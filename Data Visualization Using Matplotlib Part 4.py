# import standard libraries

import pandas as pd
import numpy as np
import csv
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats
import warnings

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

# creating df with only years columns from 1980 - 2013
df_line=df_can[years]

# Applying sum to get total immigrants year-wise
total_immigrants=df_line.sum()
total_immigrants

## LINE PLOT

fig, ax = plt.subplots()
# Plot the line
ax.plot(total_immigrants)
# Setting up the Title
ax.set_title('Immigrants between 1980 to 2013') 
# Setting up the Labels
ax.set_xlabel('Years')
ax.set_ylabel('Total Immigrants')
# Display the plot
plt.show()

# The plot function populated the x-axis with the index values (years), and the y-axis with the column values
# (population).However, notice how the years were not displayed because they are of type string.

# Create figure and axes
fig, ax = plt.subplots()
# Changing the index type to integer
total_immigrants.index = total_immigrants.index.map(int)
# Plot the line
ax.plot(total_immigrants)
# Setting up the Title
ax.set_title('Immigrants between 1980 to 2013') 
# Setting up the Labels
ax.set_xlabel('Years')
ax.set_ylabel('Total Immigrants')
# Display the plot
plt.show()

# Customizing Plot
# Create figure and axes
fig, ax = plt.subplots()
# Changing the index type to integer
total_immigrants.index = total_immigrants.index.map(int)
# Customizing the appearance of Plot
ax.plot(total_immigrants, 
        marker='s', #Including markers in squares shapes
        markersize=5, #Setting the size of the marker
        color='green', #Changing the color of the line
        linestyle="dotted") #Changing the line style to a Dotted line
# Setting up the Title
ax.set_title('Immigrants between 1980 to 2013') 
# Setting up the Labels
ax.set_xlabel('Years')
ax.set_ylabel('Total Immigrants')
ax.legend(['Immigrants'])
plt.show()

# inlcuding the background grid, a legend, and changing axis limits

# Create figure and axes
fig, ax = plt.subplots()
# Plot the line
ax.plot(total_immigrants, 
        marker='s', #Including markers in squares shapes
        markersize=5, #Setting the size of the marker
        color='green', #Changing the color of the line
        linestyle="dotted") #Changing the line style to a Dotted line
# Setting up the Title
ax.set_title('Immigrants between 1980 to 2013') 
# Setting up the Labels
ax.set_xlabel('Years')
ax.set_ylabel('Total Immigrants')
# limits on x-axis
plt.xlim(1975, 2015)  #or ax.set_xlim()
# Enabling Grid
plt.grid(True)  #or ax.grid()
# Legend
plt.legend(["Immigrants"]) #or ax.legend()
# Display the plot
plt.show()

# plot for Haiti
df_can.reset_index(inplace=True)
haiti = df_can[df_can['Country']=='Haiti']
haiti=haiti[years].transpose()
# converting the index to type integer
haiti.index = haiti.index.map(int)

# Plotting the line plot on the data
fig, ax = plt.subplots()
ax.plot(haiti)
# Setting up the Title
ax.set_title('Immigrants from Haiti between 1980 to 2013') 
# Setting up the Labels
ax.set_xlabel('Years')
ax.set_ylabel('Number of Immigrants')
# Enabling Grid
# plt.grid(True)  #or ax.grid()
# Legend
plt.legend(["Immigrants"]) #or ax.legend()
ax.annotate('2010 Earthquake',xy=(2000, 6000)) # adding annotation for the 2010 earthquake
# Display the plot
plt.show()

## SCATTER PLOT
# creating scatter plot of total immigrants to canada from 1980 to 2013

total_immigrants = df_can[years].sum()
total_immigrants.index = total_immigrants.index.map(int)

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(total_immigrants.index, 
           total_immigrants, 
           marker = 'o', 
           s = 20, # markersize = s for scatter
           color = 'darkblue')

plt.title("Immigrants between 1980 to 2013")
ax.set_xlabel("Years")
ax.set_ylabel("Total Immigrants")
plt.grid(True)
ax.legend(["Immigrants"], loc = 'upper center')
plt.show()

## BAR PLOT

# creating a bar plot to visualize the top 5 countries that contributed the most Immigrants to Canada
df_can
df_sorted = df_can.sort_values(by = "Total", ascending = False).head()
df_sorted.reset_index

label = list(df_sorted.Country)
label[2] = 'UK'

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(label, 
       df_sorted['Total'])
plt.title("Top 5 Countries with the Most Immigrants from 1980 to 2013")
ax.set_xlabel("Country")
ax.set_ylabel("Total Immigrants")
plt.show()

## HISTOGRAM
# Let's find out the frequency of distributino of the number of new immigrants from the various countries
# to Canada in 2013. 

# df_hist = df_can[['Country', '2013']]
df_hist = df_can['2013']

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df_hist)
ax.set_title('New Immigrants in 2013') 
ax.set_xlabel('Number of Immigrants')
ax.set_ylabel('Number of Countries')
ax.legend(['Immigrants'])
plt.show()

## PIE CHART
# creatign pie chart for immigration from 1980 to 1984
fig,ax=plt.subplots()
# Pie on immigrants
ax.pie(total_immigrants[0:5], labels=years[0:5], 
       colors = ['gold','blue','lightgreen','coral','cyan'],
       autopct='%1.1f%%',explode = [0,0,0,0,0.1]) #using explode to highlight the lowest 
ax.set_aspect('equal')  # Ensure pie is drawn as a circle
plt.title('Distribution of Immigrants from 1980 to 1985')
# plt.legend(years[0:5]), include legend, if you donot want to pass the labels
plt.show()

## SUB-PLOTTING

# try and use different style
print(plt.style.available)
mpl.style.use(['ggplot'])

# creating different subplots on total immigrants

fig, axs = plt.subplots(2, 2, sharey=False)
#Plotting in first axes - the left one
axs[0,0].plot(total_immigrants)
axs[0,0].set_title("Line plot on immigrants")
axs[0,0].set_xlabel("Year")
axs[0,0].set_ylabel("Total Immigrants")

#Plotting in second axes - the right one
axs[0,1].scatter(total_immigrants.index, total_immigrants)
axs[0,1].set_title("Scatter plot on immigrants")
axs[0,1].set_xlabel("Year")
axs[0,1].set_ylabel("Total Immigrants")

#Plotting in first axes - the left one
axs[1,0].bar(total_immigrants.index, total_immigrants)
axs[1,0].set_title("Bar plot on immigrants")
axs[1,0].set_xlabel("Year")
axs[1,0].set_ylabel("Total Immigrants")

#Plotting in second axes - the right one
axs[1,1].boxplot(total_immigrants)
axs[1,1].set_title("Boxplot on immigrants")
axs[1,1].set_xlabel("For all years")
axs[1,1].set_ylabel("Total Immigrants")

#Adding a Title for the Overall Figure
fig.suptitle('Subplotting Example', fontsize=15)

# Adjust spacing between subplots
fig.tight_layout()
# fig.supylabel("Total Immigrants", fontsize = 10)

# Show the figure
plt.show()

