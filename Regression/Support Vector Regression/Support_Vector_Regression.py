# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 00:36:24 2019

@author: Parth
"""

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

# =============================================================================
# I have commented the feature scaling code and haven't executed it because in Linear 
# Regression models the feature scaling is done implicitly by the library. But in some uncommon 
# Regression models like SVR, etc. we need to explicitly scale the features as
# the features aren't implicitly scaled by the library. 
# =============================================================================


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y= sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor= SVR(kernel='rbf')
regressor.fit(X,y)





plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print("r2 score:",r2_score(y,regressor.predict(X)))
