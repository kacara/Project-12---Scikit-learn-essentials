#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:02:14 2019

@author: caser
"""

#   1 import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#   2 change items
filename = 'ESN 336106, LM6000 PH (DLE 2.1), no1.csv'


#   3 variable list


#   4 import data
df1 = pd.read_csv(filename)
# summarize data
df1_shape = df1.shape
df1_head = df1.head(50)
df1_describe = df1.describe()
df1_columns = pd.Series(df1.columns)


#   5 merge to another data


#   6 prepare the data
# remove first 2 rows
df1.drop([0, 1], axis=0, inplace=True)
# remove first 2 columns
df1.drop(df1.columns[0:2], axis=1, inplace=True)
df1.reset_index(drop=True, inplace=True)

# convert 1st column dtype to datetime
df1['Date'] = pd.to_datetime(df1['Date'])
# convert other columns dtype to numeric
df1.iloc[:, 1:] = df1.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')  

#drop na, constant columns
df1_drop_columns = (df1.columns[df1.isna().all()].tolist() +
                    df1.columns[df1.nunique(axis=0, dropna=True) == 1].tolist()
                    )
df1.drop(df1_drop_columns, axis=1, inplace=True)
print('Dropped columns: ', df1_drop_columns)

#drop duplicate rows
df1.drop_duplicates(inplace=True)

#drop completely na rows except date column
m = df1.index[df1.iloc[:,1:].isna().all(axis=1) == True]
df1.drop(index = m, inplace=True)

#check and replace na values
m = df1.index[df1.iloc[:,1:].isna().any(axis=1)]  # mark na containing rows
print(len(m))  # count na containing rows
df1.drop(index = m, inplace=True)
#df1 = df1.fillna(method='ffill')
#from sklearn.impute import SimpleImputer
#https://scikit-learn.org/stable/modules/impute.html

#remove leading 0 from column_1
df1['column_1'] = df1['column_1'].map(str).str.lstrip('0')

# summarize data
df1_shape = df1.shape
df1_head = df1.head(50)
df1_describe = df1.describe()
df1_columns = pd.Series(df1.columns)


#   7 merge to another data

