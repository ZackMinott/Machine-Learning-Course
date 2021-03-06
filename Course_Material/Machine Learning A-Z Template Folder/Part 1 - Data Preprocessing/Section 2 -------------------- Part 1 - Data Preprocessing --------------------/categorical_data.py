# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Handling missing data: take the mean of all the columns
from sklearn.impute import SimpleImputer #used for preprocessing any data set
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
#fit imputer object to the matrix X
imputer = imputer.fit(X[:, 1:3]) #upper bound is excluded 
#replace missing data of the matrix X by the mean of the column
X[:, 1:3] = imputer.transform(X[:, 1:3])


#Encoding Categorical Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)
#Encoding the Dependent Variables
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

