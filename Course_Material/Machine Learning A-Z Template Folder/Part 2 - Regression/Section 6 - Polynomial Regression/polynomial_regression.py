# Polynomial Linear Regression
# Purpose: Predict the salaries of an interviewee based off certain given positions

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #takes all the rows and all the columns except the last one
y = dataset.iloc[:, 2].values # takes the dependent variable, i.e the last column

#Do not need to split training and test set because we have such a small amount of variables
"""
#Splitting the dataset into the Training Set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# fit Transform X into a matrix of new exponential independent variables
# automatically creates the constant b0 in the regression equation
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
# new linear regression model to include the X_poly fit
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1) #X_grid contains all the levels plus the incremented step
X_grid = X_grid.reshape((len(X_grid), 1)) #Reshapes X_grid into a matrix where num of lines is number of elements of X_grid
plt.scatter(X,y,color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'green')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]]) #Predicts the employee's salary based off a level 6.5
lin_reg.predict(np.array(6.5).reshape(-1,1)) #Other way

# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
