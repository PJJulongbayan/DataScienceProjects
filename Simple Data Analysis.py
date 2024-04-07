# import libraries as needed

import pandas as pd
import numpy as np
import requests
import warnings
import lxml #for better xml reading 
from bs4 import BeautifulSoup #for webscraping
import html #for webscraping

## load data to be used in analysis 
url1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

# read file as pandas dataframe
df = pd.read_csv(url1)

# check the first 5 rows of data 
df.head()

# add header colums
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers

# check first 10 rows of data after adding column headers 
df.head(10)

# quickly check the data types and data summary
df.columns
df.dtypes
df.describe() # add include = all as argument to include non-numeric columns
df.info()

# drop nas and replace '?'
df.replace("?", np.nan, inplace = True)
df.dropna(subset=["price"], axis = 0)

# describe length and compression-ratio columns
df[['length', 'compression-ratio']].describe()

# save file
df.to_csv("automobile.csv", index=False)

## NEXT FEW BLOCKS IS USING Laptop dataset
url2 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_base.csv"
df2 = pd.read_csv(url2)
df2.head()
headers2 = ["Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core", "Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg", "Price"]
df2.columns = headers2
df2.head()
## import sys to check python version
## print(sys.version)
df2 = df2.replace("?", np.NaN)
print(df2.columns)
print(df2.dtypes)
df2.describe(include="all")
print(df.info)

## NEXT FEW BLOCKS IS USING used cars dataset
url3 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df3 = pd.read_csv(url3)
df3.head()
headers3 = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df3.columns = headers3
df3.head()
df3.replace("?", np.nan, inplace = True)
df3.head(5)
missing_data = df3.isnull()
missing_data.head(5)

# the first line extract column names and convert them to a list from an array. 
# value counts count each categorical instance True/False for each column.
for column in missing_data.columns.values.tolist():
    # print(column)
    print (missing_data[column].value_counts())
    print("")

# the following code change datatype of the columns and replace missing with the dataset average        
avg_norm_loss = df3["normalized-losses"].astype("float").mean(axis=0)
print(f"Average of normalized-losses: {avg_norm_loss}")
df3["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

avg_bore=df3['bore'].astype('float').mean(axis=0)
df3["bore"].replace(np.nan, avg_bore, inplace=True)

avg_stroke = df3["stroke"].astype("float").mean(axis = 0)
df3["stroke"].replace(np.nan, avg_stroke, inplace = True)

avg_horsepower = df3['horsepower'].astype('float').mean(axis=0)
df3['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

avg_peakrpm=df3['peak-rpm'].astype('float').mean(axis=0)
df3['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

df['num-of-doors'].value_counts().idxmax()
df3["num-of-doors"].replace(np.nan, "four", inplace=True)

#we do not want to replace missing value for prediction variable
df3.dropna(subset=["price"], axis=0, inplace=True) 

df3.reset_index(drop=True, inplace=True)

df3[["bore", "stroke"]] = df3[["bore", "stroke"]].astype("float")
df3[["normalized-losses"]] = df3[["normalized-losses"]].astype("int")
df3[["price"]] = df3[["price"]].astype("float")
df3[["peak-rpm"]] = df3[["peak-rpm"]].astype("float")

df3.dtypes

# transforming df3
df3['city-L/100km'] = 235/df3["city-mpg"] 
df3.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)

# replace (original value) by (original value)/(maximum value)
df3['length'] = df3['length']/df3['length'].max()
df3['width'] = df3['width']/df3['width'].max()
df3['height'] = df3['height']/df3['height'].max() 

# show the scaled columns
df3[["length","width","height"]].head()

# Binning
df3[["horsepower"]] = df3[["horsepower"]].astype("float")
bins = np.linspace(min(df3["horsepower"]), max(df3["horsepower"]), 4)
group_names = ['Low', 'Medium', 'High']
df3['horsepower-binned'] = pd.cut(df3['horsepower'], bins, labels=group_names, include_lowest=True )
df3[['horsepower','horsepower-binned']].head(20)

# so that pandas show all columns
pd.set_option('display.max_columns', None)
df3.head()

dummy_variable_1 = pd.get_dummies(df3["fuel-type"])
dummy_variable_1.head()
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()

# merge data frame "df3" and "dummy_variable_1" and merging column
df3 = pd.concat([df3, dummy_variable_1], axis=1)
df3.drop("fuel-type", axis = 1, inplace=True)

dummy_variable_2 = pd.get_dummies(df3["aspiration"])
dummy_variable_2.head()
dummy_variable_2.rename(columns = {"std": "aspiration-std", "turbo":"aspiration-turbo"}, inplace=True)

# practice using a newly created list
list_temp = [1,2,3,4,5,6,7,8,9,10]
list_temp = pd.DataFrame(list_temp)
list_name = ["Numbers"]
list_temp.columns = list_name
list_temp_bin = np.linspace(1, 10, 4)
list_temp_bin_names = ['Low', 'Medium', 'High']
numbers_grouped = pd.cut(list_temp['Numbers'], list_temp_bin, labels = list_temp_bin_names, include_lowest= True)

dummy_variable_3 = pd.get_dummies(df3[['aspiration', 'fuel-type']])
new_df = pd.concat([df3[['aspiration', 'fuel-type']], dummy_variable_3], axis = 1)
new_df