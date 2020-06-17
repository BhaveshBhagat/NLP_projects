import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re


dataset = pd.read_csv('C:/Users/Bhagat/Documents/Python/ML/DataSets/spam_ham_dataset.csv')

ps = PorterStemmer()
corpus = []

for i in range(0 , len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ' , dataset['text'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

y = dataset['label_num']

print(X)


from sklearn.model_selection import train_test_split

X_train , X_test , Y_train, Y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(X_train , Y_train)

ypred = spam_detect_model.predict(X_test)


from sklearn.metrics import confusion_matrix ,  accuracy_score

print(confusion_matrix(Y_test , ypred),'\n',accuracy_score(Y_test,ypred))
