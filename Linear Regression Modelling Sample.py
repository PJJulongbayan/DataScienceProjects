# This lab demonstrates linear regression techniques to predict car price using the cars dataset. 

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

# import sklearn for linear regression
from sklearn.linear_model import LinearRegression as LR

# get dataset

file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
df = pd.read_csv(file_path)

# we want to show all columns when viewing the data
pd.set_option('display.max_columns', None)
df.head()
df.describe()
df.info
df.dtypes

## Simple Linear Regression
# create the lr object
lm = LR()
lm

# Using simple linear regression, we will create a linear function 
# with "highway-mpg" as the predictor variable and the "price" as the response variable.
# predictor variable should be a dataframe while response is series
lm.fit(df[['highway-mpg']], df["price"])

# trying prediction output using array of numbers generated in numpy
lm.predict(np.arange(1,101,1).reshape(-1,1))

# check intercept value and coefficient
lm.intercept_
lm.coef_

## Multiple Linear Regression
# use the following predictor variables: horsepower, curb-weight, engine-size, and highway-mpg

lm.fit(df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]], df["price"])

# check intercept value and coefficient
lm.intercept_
lm.coef_

# check model mse and r-squared
from sklearn.metrics import r2_score, mean_squared_error

mean_squared_error(df['price'], lm.predict(df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]))
r2_score(df['price'], lm.predict(df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]))

## Model Evaluation Using Visualization
# use of regression plot for single variables

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

# Let's compare this plot to the regression plot of "peak-rpm".
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

sns.residplot(x="peak-rpm", y="price", data=df)
# does not seem any direct relationship is evident. 


# for multiple linear regression, visualization is complicated using regression or residual plot 
# one way to look at the model fit is by distribution plot. 
Y_hat = lm.predict(df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]])
plt.figure(figsize=(width, height))
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.show()
plt.close()

# can use kdeplot, histplot, or displot as sns.distplot will be deprecated. 

## Polynomial regression of a single predictor
# trying for highway-mpg and price
x = df['highway-mpg']
y = df['price']

f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
# poly1d creates the polynomial form for evaluation 

# using the function below to plot the data
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
    
PlotPolly(p, x, y, 'highway-mpg')

np.polyfit(x, y, 3)

## Polynomial regression of multiple predictors

# import library and instantiate object
from sklearn.preprocessing import PolynomialFeatures as PF
pr=PF(degree=2)
pr

# define pred variables 
Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]

# fit
Z_pr=pr.fit_transform(Z)

# check shape
Z.shape
Z_pr.shape
# there are 201 samples and 4 features in original data, 15 features in transformed feature. 

## import pipeline for data processing 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# We create the pipeline by creating a list of tuples
# including the name of the model or estimator and its corresponding constructor.
Input=[('scale',StandardScaler()), ('polynomial', PF(include_bias=False)), ('model',LR())]

# We input the list as an argument to the pipeline constructor:
pipe=Pipeline(Input)
pipe

# First, we convert the data type Z to type float to avoid conversion warnings
# that may appear as a result of StandardScaler taking float inputs.
# Then, we can normalize the data, perform a transform and fit the model simultaneously.

Z = Z.astype(float)
pipe.fit(Z,y)

# Similarly, we can normalize the data, perform a transform and produce a prediction simultaneously.
ypipe=pipe.predict(Z)
ypipe[0:100]

# check metrics
mean_squared_error(df["price"], ypipe)
r2_score(df["price"], ypipe)

## OTHER MODEL EVALUATIONS

# 1. Simple Regression
lm.fit(df[['highway-mpg']], df["price"])
print('The R-square is: ', lm.score(df[['highway-mpg']], df["price"]))
# We can say that ~49.659% of the variation
# of the price is explained by this simple linear model "horsepower_fit".

# Let's calculate the MSE:
Yhat=lm.predict(df[['highway-mpg']])
print('The output of the first four predicted value is: ', Yhat[0:4])
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

# 2. Multiple Regression
Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))
# We can say that ~80.896 % of the variation of price 
# is explained by this multiple linear regression "multi_fit".

Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

# Polynomial Regression
mean_squared_error(df["price"], ypipe)
r2_score(df["price"], ypipe)

## Conclusion 
# MSE: The MSE of SLR is 3.16x10^7 while MLR has an MSE of 1.2 x10^7. The MSE of MLR is much smaller.
# R-squared: In this case, we can also see that there is a big difference between the R-squared of the SLR and the R-squared of the MLR. 
# The R-squared for the SLR (~0.497) is very small compared to the R-squared for the MLR (~0.809).

# MSE: The MSE for the MLR is smaller than the MSE for the Polynomial Fit.
# R-squared: The R-squared for the MLR is also much larger than for the Polynomial Fit.

# Comparing these three models, we conclude that the MLR model is the best model to be able to predict price
# from our dataset. This result makes sense since we have 27 variables in total and we know that more 
# than one of those variables are potential predictors of the final car price.

