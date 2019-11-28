"""
Created on Fri Nov 29 00:47:45 2019

@author: Parth
"""

'''
Concept: 

The equation of simple linear regression is: y=mx+c
Linear Regression assumes a linear relationship between a the variables 'x' and a single variable 'y'.

The regression line is plotted and the equation of the line is y=W0+W1x,
where W0 and W1 are the parameters of the model and x is the input parameter given to the model. 
Then the value of y is calculated. 


'''



import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import pandas as pd
from statistics import mean
df=pd.read_csv('Salary_Data.csv')
df.fillna(df.mean(),inplace=True)

x=df['YearsExperience'].values
y=df['Salary'].values

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

# =============================================================================
# x=np.array([1,2,3,4,5],dtype=np.float64)
# y=np.array([5,4,5,6,1],dtype=np.float64)
# 
# =============================================================================
def best_fit_slope(x,y):
    m= ((mean(x)*mean(y))-mean(x*y))/((mean(x)*mean(x))-mean(x*x))
    b= (mean(y)-m*mean(x))
    return m,b

m,b=best_fit_slope(X_train,y_train)
def predict(m,x,b):
    y=(m*x)+b
    return y

pred_values=[predict(m,i,b) for i in X_test]

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, pred_values, color = 'blue')
plt.title('Salary vs Experience (Full set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()






#X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
regressor=LinearRegression()
X_train=X_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)

regressor.fit(X_train,y_train)
X_test=X_test.reshape(-1,1)
y_pred=regressor.predict(X_test)
print(regressor.predict(pred))
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
y_test=y_test.reshape(-1,1)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


'''
