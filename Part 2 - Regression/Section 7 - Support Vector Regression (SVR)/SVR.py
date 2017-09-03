# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 21:50:20 2017

@author: Shivam
"""

#svr(not linear regressors)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values  #so that X is always a matrix of feartures and not a vector
y=dataset.iloc[:,2].values

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

#svr regressor
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)

y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


#visualise svr results
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Salary vs position')
plt.ylabel('salary')
plt.xlabel('position')

#point for ceo not included in this as for svr model, having internal penalty parameters in 
#  its algorithm and thus considers the last point to be out of bound and models itself
#  on the reamining points