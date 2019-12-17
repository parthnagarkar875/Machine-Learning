# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:33:38 2019

@author: Parth
"""

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm


df=load_boston()
boston=pd.DataFrame(df.data,columns=df.feature_names)
boston['MDEV']=df.target

X=boston.iloc[:,:-1].values
y=boston.iloc[:,13].values




X=np.append(arr=np.ones((506,1)),values=X,axis=1)
#Backward elimination starts here.
X_opt=X
regressor=sm.OLS(endog=y,exog=X_opt).fit()
regressor.summary()

X_opt=X[:,[0,1,2,3,4,5,6,8,9,10,11,12]]
regressor=sm.OLS(endog=y,exog=X_opt).fit()
regressor.summary()

X_opt=X[:,[0,1,2,4,5,6,8,9,10,11,12]]
regressor=sm.OLS(endog=y,exog=X_opt).fit()
regressor.summary()


X_opt=X_opt[:,1:]
X_train,X_test, y_train,y_test=train_test_split(X_opt,y,test_size=0.2)
#X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.2)

regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

print("R-Squared error is: ",r2_score(y_test,y_pred))
