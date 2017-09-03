# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 20:17:48 2017

@author: Shivam
"""

#polynomial regession(not linear regressors)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values  #so that X is always a matrix of feartures and not a vector
y=dataset.iloc[:,2].values

#fitting linear regression to dataset(only as a reference, no actual use here)
from sklearn.linear_model import LinearRegression
linear_regressor=LinearRegression()
linear_regressor.fit(X,y)

#fitting polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor=PolynomialFeatures(degree=4)
X_poly=polynomial_regressor.fit_transform(X)  #automatically adds the constant which we earlier added manually
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#visualise linear results
plt.scatter(X,y,color='red')
plt.plot(X,linear_regressor.predict(X),color='blue')
plt.title('Salary vs position')
plt.ylabel('salary')
plt.xlabel('position')

#visualise polynomial results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(polynomial_regressor.fit_transform(X)),color='blue')
plt.title('Salary vs Experiance (poly)')
plt.ylabel('salary')
plt.xlabel('experiance')

#predicting result (linear)
linear_regressor.predict(6.5)

#predicting result(polynomial)
lin_reg2.predict(polynomial_regressor.fit_transform(6.5))
