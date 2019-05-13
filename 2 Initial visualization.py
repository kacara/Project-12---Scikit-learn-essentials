#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:56:15 2019

@author: caser
"""

#   1 import libraries
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from pandas.plotting import parallel_coordinates

#    2 plot 3 columns including line, histogram, scatter
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

#    3 single plots

#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#scatter plot matrix --> visual corrolation
plt.style.use('seaborn')
scatter_matrix(dataset)
plt.show()

#parallel_coordinates --> visual clustering
parallel_coordinates(dataset, 'class')
plt.show()












