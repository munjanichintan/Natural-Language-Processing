# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:34:45 2020

@author: Chintan Munjani
"""
# import data

import pandas as pd

messages = pd.read_csv(r'C:\Users\Lenovo\Desktop\NLP\smsspamcollection/SMSSpamCollection', sep = '\t', names = ['label','message'])

# data process and cleaning

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lm = WordNetLemmatizer()
corpus = []

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# create bag of words model.
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 4000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y1 = y.iloc[:,1].values

# train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size = 0.20, random_state = 0)

# train model using naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)

# get a confusion matrix
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)

# get a accuracy of model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
 
