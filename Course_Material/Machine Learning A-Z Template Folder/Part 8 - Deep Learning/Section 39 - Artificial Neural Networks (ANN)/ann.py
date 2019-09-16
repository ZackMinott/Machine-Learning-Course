# Artificial Neural Network

#Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

#Installing TensorFlow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.11/get_started

# Installing Keras 
# pip install --upgrade keras

# TensorFlow and Theano now come with Keras 

# Part 1 - Data Preprocessing

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # Independent Variables
y = dataset.iloc[:, 13].values

# Encoding Categorical Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1,2])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)
#Encoding the Dependent Variables
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
X = X[:, 1:]

#Splitting the dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # Used to initialize neural network
from keras.layers import Dense # import layers for neural network

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))

# Add the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 -- Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # returns true if greater than 0.5

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)