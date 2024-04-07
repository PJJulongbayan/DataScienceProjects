# This uses basic data analysis in Python using the insurance dataset. 

# import standard libraries
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
import warnings

# import sklearn libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

## IMPORTING DATASET

# This is a filtered and modified version of the Medical Insurance Price Prediction dataset, 
# available under the CC0 1.0 Universal License on the Kaggle website.

file_path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
df = pd.read_csv(file_path, header = None)

# we want to show all columns when viewing the data
pd.set_option('display.max_columns', None)

# view dataframe
df.head(10)

# add column names
columns = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df.columns = columns
df.head(10)

# check data
df.describe()
df.dtypes

# replace '?' with Nan
df.replace('?', np.nan, inplace = True)
df

# no null values upon checking 
for cols in df.columns.to_list():
    nulls_count = df[cols].isnull().sum()
    print(f"Number of null data in {cols}: {nulls_count}")

## DATA WRANGLING

# alternate way to check for nulls or Nan
print(df.info())

# Handle missing data as follows
    # For continuous attributes (e.g., age), replace missing values with the mean.
    # For categorical attributes (e.g., smoker), replace missing values with the most frequent value.
    # Update the data types of the respective columns.
    # Verify the update using df.info().

# smoker is a categorical attribute, replace with most frequent entry
is_smoker = df['smoker'].value_counts().idxmax()
df["smoker"].replace(np.nan, is_smoker, inplace=True)

# age is a continuous variable, replace with mean age
mean_age = df['age'].astype('float').mean(axis=0)
df["age"].replace(np.nan, mean_age, inplace=True)

# Update data types
df[["age","smoker"]] = df[["age","smoker"]].astype("int")

print(df.info())

# updating charges column by rounding off digits to 2 decimal places
df[["charges"]] = np.round(df[["charges"]],2)
print(df.head())

## EXPLORATORY DATA ANALYSIS (EDA)

# Implement the regression plot for charges with respect to bmi.
sns.regplot(x= df["bmi"], y = df["charges"], data = df, line_kws={"color": "red"})
plt.ylim(0,)

# Implement the box plot for charges with respect to smoker.
sns.boxplot(x= df["smoker"], y = df["charges"], data = df)
plt.ylim(0,)

# print the correlation matrix for the dataset
print(df.corr())
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

## MODEL DEVELOPMENT
#  create a linear regression model using smoker as predictor

X = df[['smoker']]
Y = df['charges']
lm = LinearRegression()
lm.fit(X,Y)
print(lm.score(X, Y))
mean_squared_error(Y, lm.predict(X))

# create a liner regression model using all variables as predictor of charges
Z = df[["age", "gender", "bmi", "no_of_children", "smoker", "region"]]
lm.fit(Z,Y)
print(lm.score(Z, Y))
mean_squared_error(Y, lm.predict(Z))

# Creating a training pipeline that uses StandardScaler(), PolynomialFeatures() and LinearRegression() 
# to create a model that can predict the charges value using all the other attributes of the dataset.
# pipeline lis a list of tuple, can include degree as PolynomialFeatures(degree=n)
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe=Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)
print(r2_score(Y,ypipe))
mean_squared_error(Y, pipe.predict(Z))

## MODEL REFINEMENT

# Splitting the data into training and testing subsets,
# assuming that 20% of the data will be reserved for testing.
x_train, x_test, y_train, y_test = train_test_split(Z, Y, test_size = 0.2, random_state=1) 

# Initialize a Ridge regressor that used hyperparameter alpha = 0.1
# Fit the model using training data data subset and get the score on the test set. 
RR = Ridge(alpha=0.1)
RidgeModel = RR.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test,yhat))
print(mean_squared_error(y_test, yhat))

# Applying polynomial transformation to the training parameters with degree=2.
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
print(r2_score(y_test,y_hat))









