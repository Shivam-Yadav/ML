# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 00:41:10 2017

@author: Shivam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values  #we take all the lines(rows)[:], and all but last column [:-1]

#using elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]   #within cluster sum of squares
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++', max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  #automatically computes wcss value

plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel("no of clusters")
plt.ylabel("wcss")

#applying kmeans to mall dataset
kmeans=KMeans(n_clusters=5,init='k-means++', max_iter=300,n_init=10,random_state=0)
ykmeans=kmeans.fit_predict(X)

#visualising clusters
plt.scatter(X[ykmeans==0,0],X[ykmeans==0,1],s=100,c='red',label='(kanjoos log)')
plt.scatter(X[ykmeans==1,0],X[ykmeans==1,1],s=100,c='green',label='(middle class)')
plt.scatter(X[ykmeans==2,0],X[ykmeans==2,1],s=100,c='blue',label='(ameerzaade)')
plt.scatter(X[ykmeans==3,0],X[ykmeans==3,1],s=100,c='magenta',label='(dikhawati)')
plt.scatter(X[ykmeans==4,0],X[ykmeans==4,1],s=100,c='cyan',label='(gareeb)')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroid')
plt.title("Cluster of clients")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()