# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 07:52:12 2023

@author: Admin
"""

# importing the librarys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=1/3, shuffle=False)

# Fitting simple linear regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set result
y_pred =regressor.predict(X_test)

# visualising the Training set result
plt.scatter(X_train, y_train)
plt.plot(X_train, regressor.predict(X_train))
plt.title('Salary vs Experience(Training Set')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

# visualising the test set result
plt.scatter(X_test, y_test)
plt.plot(X_test, regressor.predict(X_test))
plt.title('Salary vs Experience(Training Set')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()
