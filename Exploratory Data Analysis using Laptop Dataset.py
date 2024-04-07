# This lab demonstrates exploratory data analysis using the laptop dataset

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

file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
df = pd.read_csv(file_path)

# we want to show all columns when viewing the data
pd.set_option('display.max_columns', None)
df.head()
df.describe()
df.info
df.dtypes
# some categorical values such as category and GPU appear to be categorical but with int64 dtypes. 

# %matplotlib inline as we want to view all plots in the same notebook

# Generate regression plots for each of the parameters "CPU_frequency","Screen_Size_inch" and "Weight_pounds"
# against "Price".  Also, print the value of correlation of each feature with "Price".

sns.regplot(x = "CPU_frequency", y = "Price", data = df)
sns.regplot(x = "Screen_Size_inch", y = "Price", data = df)
sns.regplot(x = "Weight_pounds", y = "Price", data = df)

# Correlation values of the three attributes with Price
for param in ["CPU_frequency", "Screen_Size_inch","Weight_pounds"]:
    print(f"Correlation of Price and {param} is ", df[[param,"Price"]].corr())

# Generate Box plots for the different feature that hold categorical values. 
# These features would be "Category", "GPU", "OS", "CPU_core", "RAM_GB", "Storage_GB_SSD"    
sns.boxplot(x = "Category", y = "Price", data = df)
sns.boxplot(x = "GPU", y = "Price", data = df)
sns.boxplot(x = "OS", y = "Price", data = df)
sns.boxplot(x = "CPU_core", y = "Price", data = df)
sns.boxplot(x = "RAM_GB", y = "Price", data = df)
sns.boxplot(x = "Storage_GB_SSD", y = "Price", data = df)

# Generate the statistical description of all the features being used in the data set.
# Include "object" data types as well.
print(df.describe())
print(df.describe(include=['object']))

# Group the parameters "GPU", "CPU_core" and "Price"
# to make a pivot table and visualize this connection using the pcolor plot.

df_group = df[["GPU", "CPU_core", "Price"]]
df_group = df_group.groupby(["GPU", "CPU_core"],as_index=False).mean()
df_group

df_group_pivot = df_group.pivot(index = "GPU", columns="CPU_core")
df_group_pivot

# let's count the unique values of each categorical values
categorical_cols = ["Manufacturer", "Category", "GPU", "OS", "CPU_core", "RAM_GB", "Storage_GB_SSD"]
for cols in categorical_cols:
    print(df[cols].value_counts()) 

# Creating plot

fig, ax = plt.subplots()
im = ax.pcolor(df_group_pivot, cmap='RdBu')

#label names
row_labels = df_group_pivot.columns.levels[1]
col_labels = df_group_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(df_group_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(df_group_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

fig.colorbar(im)

# Pearson correlation
vars = ['RAM_GB','CPU_frequency','Storage_GB_SSD','Screen_Size_inch','Weight_pounds','CPU_core','OS','GPU','Category']
for param in vars:
    pearson_coef, p_value = stats.pearsonr(df[param], df['Price'])
    print(param)
    print("The Pearson Correlation Coefficient for ",param," is", pearson_coef, " with a P-value of P =", p_value)