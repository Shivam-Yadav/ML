# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 20:38:58 2017

@author: Shivam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values 

#using dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))  #ward=uses the Ward variance minimization algorithm

plt.title('dendogram')
plt.xlabel('customers')
plt.ylabel('Euclidean Distance')
plt.show()

#fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,linkage='ward',affinity='euclidean')
y_hc=hc.fit_predict(X)

#visualising clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='(kanjoos log)')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='green',label='(middle class)')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='blue',label='(ameerzaade)')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='magenta',label='(dikhawati)')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='cyan',label='(gareeb)')

plt.title("Cluster of clients")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()

 