# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:59:24 2019

@author: Parth
"""

'''
Random Forest Regression.
--->Random Forest Regression comes under a set of learning algorithms called as Ensemble Learning.

--->Ensemble Learning is the process of taking multiple algorithms or the same algorithm multiple 
    times to generate something more powerful than the original algorithm. 
    
--->Steps in Random Forest Regression are:
    1) Select any random K points from the dataset. 
    2) Create a Decision tree to predict the dependent variable y on those k data points. 
    3) Specify the number of decision trees to be constructed. 
    4) Repeat steps 1&2 once the number of decision trees are specified in Step 3. 
    5) Let all the Decision Trees predict the dependent variable Y and the end result is computed by 
       taking the average of the results of all the Decision Trees. 
       
--->Ensemble algorithms are more stable as any changes that might affect a single tree do not affect
    the end outcome as the average of the outcomes of all trees is taken. 
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


from sklearn.ensemble import RandomForestRegressor
regressor= RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X,y)


y_pred=regressor.predict(np.array([[6.5]]))

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print("r2 score:",r2_score(y,regressor.predict(X)))
