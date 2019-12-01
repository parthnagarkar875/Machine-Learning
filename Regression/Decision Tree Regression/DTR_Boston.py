# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:22:02 2019

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


y=y.reshape(-1,1)

X=np.append(arr=np.ones((506,1)),values=X,axis=1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y= sc_y.fit_transform(y)

#Backward elimination starts here.
X_opt=X
regressor=sm.OLS(endog=y,exog=X_opt).fit()
regressor.summary()

X_opt=X[:,[0,1,2,4,5,6,7,8,9,10,11,12]]
regressor=sm.OLS(endog=y,exog=X_opt).fit()
regressor.summary()

X_opt=X_opt[:,1:]
X_train,X_test, y_train,y_test=train_test_split(X_opt,y,test_size=0.2)


from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)


print("R-Squared error is: ",r2_score(y_test,regressor.predict(X_test)))
