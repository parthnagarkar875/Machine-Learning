# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 22:55:00 2020

@author: Parth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df=pd.read_csv('Customer_data.csv')
X=df.iloc[:,3:13].values
y=df.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,1]=labelencoder.fit_transform(X[:,1])
labelencoder2=LabelEncoder()
X[:,2]=labelencoder2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()

X=X[:,1:]

sc_X=StandardScaler()
X=sc_X.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)