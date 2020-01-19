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

df=pd.read_csv('Churn_Modelling.csv')
df2=pd.get_dummies(df['Gender'])
df3=pd.get_dummies(df['Geography'])

result = pd.concat([df,df2,df3],axis=1)
df=result.drop(['Geography','Gender'],axis=1)

y=df.iloc[:,11].values

X=df.iloc[:,[3,4,5,6,7,8,9,10,12,13,14,15,16]].values


sc_X=StandardScaler()
X=sc_X.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()




