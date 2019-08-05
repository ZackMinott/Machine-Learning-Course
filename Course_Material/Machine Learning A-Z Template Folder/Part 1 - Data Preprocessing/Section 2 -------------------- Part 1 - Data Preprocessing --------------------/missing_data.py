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