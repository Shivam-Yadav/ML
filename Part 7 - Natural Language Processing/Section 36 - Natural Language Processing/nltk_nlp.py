# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 23:37:49 2017

@author: Shivam
"""
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
#quoting=3 to ignore the double quotes "
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#cleaning text
import re
#import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')   to download this 'stopwords'
#we need to spit the review into different words i.e list for running through the words from stopwords
#stemming (getting the root word eg loved-love, so we can reduce sparse matrix)
from nltk.stem.porter import PorterStemmer #for stemming
corpus = []  #new list of cleaned reviews
for i in range(0, 1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i],) #we dont want to remove these characters
    #the removed characters are replaced by 'space'
    review=review.lower()
    review=review.split()  #converts to a list
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #only include words not in 'stopwords(contains words in many languages)'
    #we can use without using 'set' but using it makes the result achieving faster as there are algo that go much faster in a set than in a list
    review = ' '.join(review)  #makes the review string again
    corpus.append(review)

#creating bag of words model (minimize no of words and create a sparse matrix)
from sklearn.feature_extraction.text import CountVectorizer    #we can do clearing text here but, doing it seperately gives us more control
cv=CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()   #we use max_features in countvectorizer to get words whic appear max times
y = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

