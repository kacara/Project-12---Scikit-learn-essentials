#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 09:10:39 2018

@author: caser
"""

# 1 import libraries
#import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import datetime
#import seaborn as sns
#from scipy import stats
#from scipy import io as spio
from sklearn.datasets import load_boston
#from sklearn.linear_model import Lasso
#from sklearn.metrics import r2_score


# variable list
mule = []
m = []
filename = []
df1 = []  # imported data
df2 = []  # ML data no1
df3 = []  # ML data no2
df1_shape = []
df1_head = []
df1_describe = []
df1_columns = []
PT = 0.9  # percent of trained data over complete data, rest is test data
X_train = []
y_train = []
X_test = []
y_test = []


# 2 import data
boston = load_boston()
#filename = 'ESN 336106, LM6000 PH (DLE 2.1), no1.csv'
#df1 = pd.read_csv(filename)


# 4 create a pandas version of data 
df1 = pd.DataFrame(boston['data'], columns=boston['feature_names'])
df1['target_MEDV'] = pd.DataFrame(boston['target'])

#df1_head = df1.head
#df1_describe = df1.describe()
#df1_columns_types = pd.DataFrame(df1.dtypes)


# 5 initial tidy up of data
#df1.drop([0, 1], axis=0, inplace=True)  # remove first 2 rows
#df1.drop(df1.columns[0:2], axis=1, inplace=True)  # remove first 2 columns
#df1.reset_index(drop=True, inplace=True)

#df1['Date'] = pd.to_datetime(df1['Date'])  # convert 1st column dtype to datetime
#df1.iloc[:, 1:] = df1.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')  # convert other columns dtype to numeric

##drop na, constant columns
#df1_drop_columns = (df1.columns[df1.isna().all()].tolist() +
#                    df1.columns[df1.nunique(axis=0, dropna=True) == 1].tolist()
#                    )
#df1.drop(df1_drop_columns, axis=1, inplace=True)
#print('Dropped columns: ', df1_drop_columns)

#drop duplicate rows
df1.drop_duplicates(inplace=True)

##drop completely na rows except date column
#m = df1.index[df1.iloc[:,1:].isna().all(axis=1) == True]  # mark na rows
#df1.drop(index = m, inplace=True)

##check and replace na values
#m = df1.index[df1.iloc[:,1:].isna().any(axis=1)]  # mark na containing rows
#print(len(m))  # count na containing rows
#df1.drop(index = m, inplace=True)
##df1 = df1.fillna(method='ffill')
##from sklearn.impute import SimpleImputer
##https://scikit-learn.org/stable/modules/impute.html

df1_head = df1.head
df1_describe = df1.describe()
df1_columns_types = pd.DataFrame(df1.dtypes)


# 6 visualize and check data as line and scatter plot vs target 
m = len(df1.columns)
fig = plt.figure(figsize=(24, m*3))
for idx, val in enumerate(df1.columns,start=0):
    plt.subplot2grid((m, 3), (idx, 0))
    plt.title('Lineplot of ' + str(val) + ' vs index') 
    plt.ylabel(val)
#    plt.xlabel('index')
    plt.grid(True)
    plt.minorticks_on()
    plt.plot(val, data=df1, linestyle ='solid', linewidth=0.2, color="0.6",
             marker='.', markersize=6, markeredgecolor='blue')
    
    plt.subplot2grid((m, 3), (idx, 1))
    plt.title(str(val) + ' histogram') 
#    plt.ylabel(val)
#    plt.xlabel('target')
    plt.grid(False)
    plt.minorticks_on()
    plt.hist(val, data=df1, ec='white')
    
    plt.subplot2grid((m, 3), (idx, 2))
    plt.title('Scatterplot of ' + str(val) + ' vs target_MEDV') 
#    plt.ylabel(val)
#    plt.xlabel('target')
    plt.grid(False)
    plt.minorticks_on()
    plt.scatter('target_MEDV', val, data=df1, alpha=0.5)

plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.show()
#plt.savefig('descriptive plots.pdf', facecolor="0.9" )

#
#        - CRIM     per capita crime rate by town
#        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#        - INDUS    proportion of non-retail business acres per town
#        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#        - NOX      nitric oxides concentration (parts per 10 million)
#        - RM       average number of rooms per dwelling
#        - AGE      proportion of owner-occupied units built prior to 1940
#        - DIS      weighted distances to five Boston employment centres
#        - RAD      index of accessibility to radial highways
#        - TAX      full-value property-tax rate per $10,000
#        - PTRATIO  pupil-teacher ratio by town
#        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#        - LSTAT    % lower status of the population
#        - MEDV     Median value of owner-occupied homes in $1000's


# 10 prepare data for ML
# scale columns
from sklearn.preprocessing import MinMaxScaler
df3 = MinMaxScaler().fit_transform(df1)  # df3 data type became a numpy array
df3_describe = pd.DataFrame(df3).describe()

# # preprocess data ( https://scikit-learn.org/stable/modules/preprocessing.html )
# normalize and feature scaling 
# x_new = (x_specific - xmin) / (xmax - xmin)
# reduce dimensionality of columns, feature selection, dimensionality reduction
# consider Heterogeneity of the data types
# reduce noise in rows by describe
# reduce repeating rows?


# 11 define a random sample as simplified data and
# fit&predict for trial and time reduction


# 12 define train & test data and algorithm

# 12a LinearRegression (scaling is not required)
# ML data creation
df2 = df1
PT = 0.9
X_train = df2.iloc[:round(df1.shape[0] * PT) ,:-1]
y_train = df2.iloc[:round(df1.shape[0] * PT) ,-1]
X_test =  df2.iloc[round(df1.shape[0] * PT): ,:-1]
y_test =  df2.iloc[round(df1.shape[0] * PT): ,-1]
# or with sklearn-random splitter for randomization
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        df2.iloc[:,:-1], df2.iloc[:,-1], test_size=0.1, random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# fitting
reg = LinearRegression().fit(X_train, y_train)
r_square1 = reg.score(X_train, y_train)  # coeff. of determination
coef1 = reg.coef_
interc_1 = reg.intercept_
# predicting and scoring
y_pred1 = reg.predict(X_test)
score1 = r2_score(y_test, y_pred1)

# 12b LinearRegression (scaling is not required)






# 12b optimize with Hyper-parameters 
clf = SVC()
clf.set_params(kernel='linear').fit(X, y).predict(X_test)
clf.set_params(kernel='rbf', gamma='scale').fit(X, y).predict(X_test)


