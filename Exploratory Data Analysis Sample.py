# import libraries
import pandas as pd
import numpy as np
import requests
import json
import csv
import html
from bs4 import BeautifulSoup
import seaborn as sns
from matplotlib import pyplot as plt
import stats

# %matplotlib inline -- because want to display results in the notebook; similar to SQL magic. 
x = [1,2,3]
y = [4,5,6]
df = pd.DataFrame({"X":x, "Y":y})

# basic line chart between x and y
plt.plot(x,y)
# basic scatter chart
plt.scatter(x,y)
# histogram, need to create bins (default is 10)
plt.hist(x, bins = 10)
# bar chart (x needs to be categorical data)
plt.bar(x,y)
# pseudocolor plot displays matrix data as an array of colored cells (known as faces)
# has to be a pivot table
# plt.pcolor(C)

# regression
sns.regplot(x= "X", y = "Y", data = df)
# boxplot
sns.boxplot(x= "X", y = "Y", data = df)
# residual plot
sns.residplot(x= "X", y = "Y", data = df)
# Kernel Density Plot (KDE)
# creates a probability distribution curve for the data based upon its likelihood of occurence on specific value
sns.kdeplot(x)
sns.kdeplot(y)
# Distribution plot (KDE +  histogram); has been deprecated 
sns.distplot(x, hist=True)

# correlation coefficient
stats.corr(x,y)
