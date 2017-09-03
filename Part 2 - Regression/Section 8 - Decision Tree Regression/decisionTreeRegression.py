# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 00:14:47 2017

@author: Shivam
"""

#decision tree regession(not linear regressors)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values  #so that X is always a matrix of feartures and not a vector
y=dataset.iloc[:,2].values

#make regressor
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
#fitting decision tree regression model
regressor.fit(X,y)

y_pred=regressor.predict(6.5)

#visualise decision tree results
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Salary vs position(decision tree)')
plt.ylabel('salary')
plt.xlabel('position')
