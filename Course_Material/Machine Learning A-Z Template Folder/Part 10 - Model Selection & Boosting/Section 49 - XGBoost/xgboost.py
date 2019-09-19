# XGBoost

# Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#

# Importing the libraries
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

# Fitting XGBoost to the Training Set 
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Apply k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean() # relevant evaluation of our model performance
accuracies.std() # tells us if there is a high or low variance
