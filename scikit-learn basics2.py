#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:58:25 2019

@author: caser
"""

import pandas
from pandas.plotting import scatter_matrix
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#Summarize the Dataset
dataset.shape
dataset.head(20)
dataset.describe()
dataset.groupby('class').size()


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# scatter plot matrix --> visual corrolation
plt.style.use('seaborn')
scatter_matrix(dataset)
plt.show()

# parallel_coordinates --> visual clustering
parallel_coordinates(dataset, 'class')
plt.show()



