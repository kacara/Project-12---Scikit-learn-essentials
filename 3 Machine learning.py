#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:36:21 2019

@author: caser
"""

#    10 prepare data for ML
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
# remove similar or related columns


#   11 define a random sample as simplified data and
# fit&predict for trial and time reduction
