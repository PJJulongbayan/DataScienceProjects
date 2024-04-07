# This lab demonstrates modelling evaluatino techniques on linear regression models using the cars dataset. 

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

warnings.filterwarnings('ignore')

# get dataset

file_path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
df = pd.read_csv(file_path)

# we want to show all columns when viewing the data
pd.set_option('display.max_columns', None)
df.head()
df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)
df.describe()
df.info
df.dtypes

## For the purposes of this exercise, let's only use the numeric data
df=df._get_numeric_data()
df.head()

## Splitting Training and Test Set

# because we need to import train_test_split
from sklearn.model_selection import train_test_split
y_data = df["price"]
x_data=df.drop('price',axis=1)

# train test split at 10% test size and seed 1
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

# train test split at 40% test size and seed 0
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0) 
print("number of test samples :", x_test1.shape[0])
print("number of training samples:",x_train1.shape[0])

# create a linear regression object
lre = LinearRegression()
# fit model using horsepower as predictor
lre.fit(x_train1[['horsepower']],y_train1)
# check R-squared on train data
lre.score(x_train1[['horsepower']],y_train1)
# check R-squared on test data
lre.score(x_test1[['horsepower']],y_test1)

# get mean square error on test data
mean_squared_error(y_test1, lre.predict(x_test1[['horsepower']]))

## Cross Validation
from sklearn.model_selection import cross_val_score, cross_val_predict

# We input the object, the feature ("horsepower"), and the target data (y_data).
# The parameter 'cv' determines the number of folds. In this case, it is 4.
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)

# We can calculate the average and standard deviation of our estimate:
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

# We can use negative squared error as a score by setting the parameter 'scoring' metric
# to 'neg_mean_squared_error'.
-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')

# predicting using cross_val
cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)

## Overfitting, Underfitting, and Model Selection
# "out of sample data", is a much better measure of how well 
# your model performs in the real world. One reason for this is overfitting.

lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

# prediction using training data
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# prediction using test data
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# Let's perform some model evaluation using our training and testing data separately.


# Create function for plotting
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()

# Let's examine the distribution of the predicted values of the training data.
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

# Let's examine the distribution of the predicted values of the test data.
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

# it is evident that the distribution of the data in Figure 1 is much better at fitting the data. 
# This difference in Figure 2 is apparent in the range of 5000 to 15,000.

# Overfitting occurs when the model fits the noise, but not the underlying process. 
# Therefore, when testing your model using the test set, your model does not perform 
# as well since it is modelling noise, not the underlying process that generated the relationship

## RIDGE REGRESSION

# Let's import Ridge from the module linear models.
from sklearn.linear_model import Ridge

# Let's create a Ridge regression object, setting the regularization parameter (alpha) to 1
RidgeModel=Ridge(alpha=1)

# using degree two polynomial transformation on our data.
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

RidgeModel.fit(x_train_pr, y_train)
yhat = RidgeModel.predict(x_test_pr)

# We select the value of alpha that minimizes the test error. To do so, we can use a for loop.
# We have also created a progress bar to see how many iterations we have completed so far.

from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RidgeModel = Ridge(alpha=alpha) 
    RidgeModel.fit(x_train_pr, y_train)
    test_score, train_score = RidgeModel.score(x_test_pr, y_test), RidgeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)
    
    
# We can plot out the value of R^2 for different alphas:
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()


## Grid Search
# The term alpha in Ridge Regression above is a hyperparameter. 
# Sklearn has the class GridSearchCV to make the process of finding the best hyperparameter simpler.

# Let's import GridSearchCV from the module model_selection.
from sklearn.model_selection import GridSearchCV

# We create a dictionary of parameter values:
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
parameters1

# Create a Ridge regression object. No need to pass an initial value of alpha. 
RR=Ridge()

Grid1 = GridSearchCV(RR, parameters1,cv=4)

# Fit the model:
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

# The object finds the best parameter values on the validation data.
# We can obtain the estimator with the best parameters and assign it to the variable BestRR as follows:

BestRR=Grid1.best_estimator_
BestRR

# we now test our model on the test data: 
BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)

# normalization parameter is not accepted anymore in Ridge