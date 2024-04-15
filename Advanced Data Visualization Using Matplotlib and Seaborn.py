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


## WAFFLE CHART

# let's create a new dataframe for these three countries 
df_dsn = df_can.loc[['Denmark', 'Norway', 'Sweden'], :]
# let's take a look at our dataframe
df_dsn

fig = plt.figure(FigureClass = Waffle,
                 rows = 20, columns = 30, #pass the number of rows and columns for the waffle 
                 values = df_dsn['Total'], #pass the data to be used for display
                 cmap_name = 'tab20', #color scheme
                 legend = {'labels': [f"{k} ({v})" for k, v in zip(df_dsn.index.values,df_dsn.Total)],
                            'loc': 'lower left', 'bbox_to_anchor':(0,-0.1),'ncol': 3}
                 #notice the use of list comprehension for creating labels 
                 #from index and total of the dataset
                )

#Display the waffle chart
plt.show()

## WORD CLOUDS

# NEW Data set - short novel written by Lewis Carroll titled Alice's Adventures in Wonderland

# open the file and read it into a variable alice_novel
import requests
import urllib

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/alice_novel.txt'
alice_novel = requests.get(url).text
# alice_novel
# alice_novel = urllib.request.urlopen(url).read().decode("utf-8")

# instantiate a word cloud object
alice_wc = WordCloud()

# generate the word cloud
alice_wc.generate(alice_novel)

# display the word cloud
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# let's resize the cloud so that we can see the less frequent words a little better.
fig = plt.figure(figsize=(14, 18))

# display the cloud
plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# said isn't really an informative word. So let's add it to our stopwords and re-generate the cloud.
# We use STOPWORDS function to remove any redundant stopwords and edit formatting of word cloud.
stopwords = set(STOPWORDS)
stopwords.add('said') 
# use update to include other words such as Alice
stopwords.update(['said', 'Alice'])
alice_wc = WordCloud(stopwords=stopwords,
                     background_color='white',  # Background color
                     max_words=2000,  # Maximum number of words
                     max_font_size=80,  # Maximum font size
                     colormap='viridis',  # Color map
                     contour_width=3,  # Contour width
                     contour_color='steelblue'  # Contour color
                     )
# re-generate the word cloud
alice_wc.generate(alice_novel)

# display the cloud
fig = plt.figure(figsize=(14, 18))

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# adding image to overlay the word cloud
#save mask to alice_mask
alice_mask = np.array(Image.open(urllib.request.urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/labs/Module%204/images/alice_mask.png')))
alice_mask

fig = plt.figure(figsize=(14, 18))

plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis('off')
plt.show()

# rendering the word cloud with mask and removing other custom config
alice_wc = WordCloud(stopwords=stopwords,
                     background_color='white',  # Background color
                     mask=alice_mask,
                     max_words=2000,  # Maximum number of words
                     )
# re-generate the word cloud
alice_wc.generate(alice_novel)

# display the cloud
fig = plt.figure(figsize=(14, 18))

plt.imshow(alice_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

## Word Cloud using immigration data
df_can
total_immigration = df_can['Total'].sum()
total_immigration

max_words = 90
word_string = ''
for country in df_can.index.values:
     # check if country's name is a single-word name
    if country.count(" ") == 0:
        repeat_num_times = int(df_can.loc[country, 'Total'] / total_immigration * max_words)
        word_string = word_string + ((country + ' ') * repeat_num_times)

# display the generated text
word_string

# create the word cloud
wordcloud = WordCloud(background_color='white').generate(word_string)

print('Word cloud created!')

# display the cloud
plt.figure(figsize=(14, 18))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

## PLOTTING WITH SEABORN

# In our data 'df_can', let's find out how many continents are mentioned
df_can['Continent'].unique()

# count plot - histogram across categorical variables
sns.countplot(x='Continent', data=df_can)
plt.title("Count of Countries by Continent")
plt.show()

# replacing long continent names
df_can1 = df_can.replace('Latin America and the Caribbean', 'L-America')
df_can1 = df_can1.replace('Northern America', 'N-America')

plt.figure(figsize=(15, 10))
sns.countplot(x='Continent', data=df_can1)
plt.title("Count of Countries by Continent")
plt.show()

# Barplot
plt.figure(figsize=(15, 10))
sns.barplot(x='Continent', y='Total', data=df_can1)
plt.show()

# Regression Plot

years = list(map(str, range(1980, 2014)))
# we can use the sum() method to get the total population per year
df_tot = pd.DataFrame(df_can[years].sum(axis=0))

# change the years to type float (useful for regression later on)
df_tot.index = map(float, df_tot.index)

# reset the index to put in back in as a column in the df_tot dataframe
df_tot.reset_index(inplace=True)

# rename columns
df_tot.columns = ['year', 'total']

# view the final dataframe
df_tot.head()

# creating regplot
sns.regplot(x='year', y='total', data=df_tot)

# modifying color to green
sns.regplot(x='year', y='total', data=df_tot, color='green')
plt.show()

# let's use + instead of circular markers
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.show()

# Let's blow up the plot a little so that it is more appealing to the sight.
plt.figure(figsize=(15, 10))
sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.show()

# Further modification
plt.figure(figsize=(15, 10))

sns.set(font_scale=1.5)

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()