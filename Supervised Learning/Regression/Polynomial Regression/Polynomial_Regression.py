# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:39:02 2019

@author: Parth
"""
'''
--->The regression line in Linear regressionn is a straight line that passes 
    through the plotted points. It works well in predicting the future points when 
    the values of the dependent variable change with the changes in the values of the 
    independent variables. 
    
--->But if the plotted points are curved, then fitting a straight line won't be 
    useful for predicting the values. To overcome this, we use Polynomial Linear 
    Regression. In Polynomial Linear regression, the equation is of the form:
    Y=B0+B1X1+B1(X1)^2+B1(X1)^3+....
    We take the polynomial power of the independent variable so that the regression 
    line forms a curve.
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

# =============================================================================
# I have commented the feature scaling code and haven't executed it because in Linear 
# Regression models the feature scaling is done implicitly by the library. But in some uncommon 
# Regression models like SVR, etc. we need to explicitly scale the features as
# the features aren't implicitly scaled by the library. 
# =============================================================================

'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
'''


X_poly=PolynomialFeatures(degree=3)
PolyReg=X_poly.fit_transform(X)

cl=LinearRegression()
cl.fit(PolyReg,y)
y_pred=cl.predict(PolyReg)


plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print("r2 score:",r2_score(y,y_pred))








