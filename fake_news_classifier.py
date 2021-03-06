# -*- coding: utf-8 -*-
"""Fake_news_classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w31Qko3CmQNX9vJBRrmDaD3v_j3Ghr5R
"""

import pandas as pd

FackNewsDB = pd.read_csv("/content/drive/My Drive/Fake_news_dataset/train.csv")
FackNewsDB

FackNewsDB = FackNewsDB.dropna()

X= FackNewsDB.drop('label' , axis=1)
X = X.reset_index()

y = FackNewsDB['label']

X.shape , y.shape

import tensorflow as tf
tf.__version__

import nltk
import re
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#using stemming
ps = PorterStemmer()


words = []
#cleaning words
for i in range(0 , len(X)):
   review = re.sub('^[a-zA-Z]',' ', X['title'][i])
   review = review.lower()
   review = review.split()
   
   review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
   review = ' '.join(review)
   words.append(review)

#using Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(words).toarray()

#divided into train test data
from sklearn.model_selection import train_test_split

X_train , X_test , Y_train, Y_test = train_test_split(X , y , test_size=0.2 , random_state=0)

#model creation
from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(X_train , Y_train)

ypred = spam_detect_model.predict(X_test)



from sklearn.metrics import confusion_matrix , accuracy_score

print('BOW \n',confusion_matrix(Y_test , ypred),'\n',accuracy_score(Y_test,ypred))

#using TFIDF 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
Xtfidf = tfidf.fit_transform(words).toarray()


X_trainTf , X_testTf , Y_trainTf, Y_testTf = train_test_split(Xtfidf , y , test_size=0.2 , random_state=0)


TFIDF_spam_detect_model = MultinomialNB().fit(X_trainTf , Y_trainTf)

Tfidf_ypred = TFIDF_spam_detect_model.predict(X_testTf)


print('TF_IDF\n',confusion_matrix(Y_testTf , Tfidf_ypred),'\n',accuracy_score(Y_testTf,Tfidf_ypred))



