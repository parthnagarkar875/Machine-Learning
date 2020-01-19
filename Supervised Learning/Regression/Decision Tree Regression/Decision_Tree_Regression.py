# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:00:01 2019

@author: Parth
"""
#https://www.saedsayad.com/decision_tree.htm
'''
--->Decision Tree algorithm constructs a tree like structure to divide the labels based 
    on the conditional values. 

--->The concept of Entropy and Information gain is used in Decision Trees. 

--->Entropy is a measure of the IMPURITY present in the data. We initially calculate the entropy of 
    the target variable and then subtract the entropy from the entropy of eah feature. 
    The subtracted value then obtained is called as Information Gain. 

--->The feature with the highest information gain is placed is placed as the child of the root node. 
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_csv('Position_Salaries.csv')
X=df.iloc[:,1].values
y=df.iloc[:,2].values

X=X.reshape(-1,1)
y=y.reshape(-1,1)

'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y= sc_y.fit_transform(y)
'''

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


y_pred=regressor.predict(np.array([[6.5]]))


plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print("r2 score:",r2_score(y,regressor.predict(X)))
