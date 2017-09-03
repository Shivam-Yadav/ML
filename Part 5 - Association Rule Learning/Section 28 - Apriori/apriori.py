# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 00:34:06 2017

@author: Shivam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Market_Basket_Optimisation.csv",header=None)

#apriori expects data input to be in format of list of lists in form of string
transactions=[]
for i in range (0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#training apriori on dataset
from apyori import apriori
#we do for products purchased 3 times a day, so 7*3/7500=0.0028 for week min_support
#high min_confidence gives obvious answers
rules=apriori(transactions,min_support=0.003,min_confidence= 0.2,min_lift=3,min_length=2)

#visualising results
results=list(rules)