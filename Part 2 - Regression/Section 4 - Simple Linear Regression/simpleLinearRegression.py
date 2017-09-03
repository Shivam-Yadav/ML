# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:30:17 2017

@author: Shivam
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=1/3)

#fitting simple linear regression to training set

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test results

y_pred=regressor.predict(X_test)

#visualise set

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experiance (Training set)')
plt.ylabel('salary')
plt.xlabel('experiance')

#visualise test set
plt.scatter(X_test,y_test,color='green')
plt.show()