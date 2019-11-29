# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:07:26 2019

@author: Parth
"""

'''
Multiple Linear Regression.

--->In Simple Linear Regression, we had just one independent variable. Based on the value 
    of that variable, we would predict the value of the dependent variable. 
    
--->But the real world scenarios are so complex that is it quite impossible to predict a 
    continuous value based on just one single feature. So when we try to predict a continuous value
    based on a set of multiple features given as inputs, that regression is called as Multiple 
    Linear Regression. 
    
--->Multiple Linear Regression's equation is of the form y=B0+B1X1+B2X2+B3X3+BnXn
    Where, 
    X1,X2 are the input features. 
    
--->In Multiple Linear Regression, we should make sure that the features are correlated with 
    the dependent variabele and not with each other i.e Multicollinearity must not exist. 
'''
from sklearn.metrics import accuracy_score

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df=pd.read_csv('headbrain.csv')
x=df.iloc[:,:-1].values
y=df.iloc[:,3].values

X_train, X_test, y_train, y_test= train_test_split(x,y,test_size=0.2)



cl=LinearRegression()
cl.fit(X_train,y_train)

y_pred=cl.predict(X_test)

ssr=0
sst=0

for i in range(len(y_test)):
    ssr+=(y_test[i]-y_pred[i])**2
    sst+=(y_test[i]-mean(y_test))**2

r2=1-(ssr/sst)


print("R-Squared error is: ",r2_score(y_test,y_pred))










