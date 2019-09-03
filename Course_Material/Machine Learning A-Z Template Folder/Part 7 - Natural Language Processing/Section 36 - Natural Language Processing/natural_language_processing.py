# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re 
import nltk
nltk.download('stopwords') # download all the words that are irrelevant or propositional from nltk library
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] # a corpus is a collection of texts that could be anything of the same type
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ' ,dataset['Review'][i]) #specify what you don't want to remove
    review = review.lower() #change all characters to lower case
    review = review.split() #splits the string into a list of different words
    ps = PorterStemmer() # stemming changes words to its root "Loved" -> "love"
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # if word doesn't exist in stopwords, keep it in review list
    review = ' '.join(review) # takes a list and converts it back into a string with spaces between strings
    corpus.append(review) # appends the cleaned review to the corpus
    
# Creating the Bag of Words Model (takes all the words of the review and creates one column for each of these words)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # converts a collection of text documents to a matrix of token counts with a max of frequent words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#Splitting the dataset into the Training Set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Naive Bayes to the Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Stats
Accuracy = (55+91)/200 # accuracy of the set yield 73 %
Precision = (91) / (42+91) 
Recall = 91 / (91 + 12)
F1 = 2 * Precision * Recall/(Precision + Recall)