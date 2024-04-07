# This lab demonstrates exploratory data analysis using the cars dataset

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
from scipy import stats

# get dataset

file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
df = pd.read_csv(file_path)

# we want to show all columns when viewing the data
pd.set_option('display.max_columns', None)
df.head()
df.describe()
df.info
df.dtypes

# %matplotlib inline as we want to view all plots in the same notebook

# lets try and find the correlation between all numeric values in the df
# df.corr() 
# view correlation on bore, stroke, compression-ratio, and horsepower
df[["bore", "stroke", "compression-ratio", "horsepower"]].corr()

# finding relationship between engine size and price
sns.regplot(x="engine-size", y = "price", data = df)
plt.ylim(0,) 
# plt.xlabel('Engine Size')
# plt.title("Scatter Plot Between Price and Engine Size")
# plt.show()

# view correlation between engine size and price
df[["engine-size", "price"]].corr()

# finding relationship between highway mpg and price
sns.regplot(x="highway-mpg", y = "price", data = df)

# let's try correlation between variable hypothesized to have weak correlation
sns.regplot(x="peak-rpm", y="price", data=df)
df[['peak-rpm','price']].corr()

# Now let's try doing analysis for CATEGORICAL VARIABLES
sns.boxplot(x="body-style", y="price", data=df)
# We see that the distributions of price between the different body-style categories have a significant overlap,
# so body-style would not be a good predictor of price. Let's examine engine "engine-location" and "price"

sns.boxplot(x="engine-location", y="price", data=df)
# Here we see that the distribution of price between these two engine-location categories, front and rear,
# are distinct enough to take engine-location as a potential good predictor of price.

sns.boxplot(x="drive-wheels", y="price", data=df)
# Here we see that the distribution of price between the different drive-wheels categories differs.
# As such, drive-wheels could potentially be a predictor of price.

# DESCRIPTIVE STATISTICAL ANALYSIS
df.describe() # all numeric
df.describe(include=['object']) # all objects

# Value counts is a good way of understanding how many units of each characteristic/variable we have. 
df['drive-wheels'].value_counts()
df['drive-wheels'].value_counts().to_frame()
# Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts" 
# and rename the column 'drive-wheels' to 'value_counts'.
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
# Now let's rename the index to 'drive-wheels':
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts

# GROUPING
# lets see all unique value of drive wheels
df['drive-wheels'].unique()
# Calculating average price of each of the different categories of data below
df_group_one = df[["drive-wheels","body-style","price"]]
df_group_one = df_group_one.groupby(["drive-wheels", "body-style"],as_index=False).mean()
df_group_one
# stats are easier to visualize as a pivot table
df_group_one_pivot = df_group_one.pivot(index='drive-wheels',columns='body-style')
df_group_one_pivot.fillna(0, inplace=True)
df_group_one_pivot

# plot using heatmap
plt.pcolor(df_group_one_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

# let's try correlation on some variables
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
# stats.pearsonr(df['wheel-base'], df['price'])
# Since the p-value is 0.001, the correlation between wheel-base and price
# is statistically significant, although the linear relationship isn't extremely strong (~0.585).

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
# strong correlation, close to 1. 

