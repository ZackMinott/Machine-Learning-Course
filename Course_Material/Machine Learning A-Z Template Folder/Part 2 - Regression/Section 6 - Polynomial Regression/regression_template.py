#Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #takes all the rows and all the columns except the last one
y = dataset.iloc[:, 2].values # takes the dependent variable, i.e the last column

#Do not need to split training and test set because we have such a small amount of variables
#Will have to use if you have more than 10 variables
"""
#Splitting the dataset into the Training Set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

#Fitting the Regression Model to the dataset
#Create your regressor here

# Predicting a new result with Polynomial Regression
y_pred = regressor.predict([[6.5]])

#Visualising the Regression results
plt.scatter(X,y,color = 'red')
plt.plot(X, regressor.predict(X), color = 'green')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Regression results (for Higher Resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()