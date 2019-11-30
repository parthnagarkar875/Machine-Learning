"""
Created on Fri Nov 29 00:47:45 2019

@author: Parth
"""

'''
-----------------------------------------Concept:-----------------------------------

--->The equation of simple linear regression is: y=mx+c
--->Linear Regression assumes a linear relationship between a the variables 'x' and a single variable 'y'.

--->The regression line is plotted and the equation of the line is y=B0+B1x,
    where B0 and B1 are the parameters of the model and x is the input parameter given to the model. 
    Then the value of y is calculated. 


--->The catch lies in determining the values of the parameters of the model: c and m (B0 and B1).

--->To determine the values, we use the following formula:
    B1= (x'.y'-(xy)')/((x'.x')-(x.x)')
    B0= y'-B1.x'


--->Where, x' and y' indicate the mean values of x and y respectively. 

--->After determining the values of B0 and B1, we just need to subsitute the input value of X in the equation.

--->After making the predictions, we need to determine the error in our predictions. 
    for that, we use the r-squared method. 
    The formula for r-squared method is: 
        r=1-(ssr/sst)
    Where, ssr= Error in regression line (y-y_pred)
           sst= Deviation in mean (y-y')

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.metrics import r2_score
import pandas as pd
from statistics import mean
df=pd.read_csv('Salary_Data.csv')
df.fillna(df.mean(),inplace=True)

x=df['YearsExperience'].values
y=df['Salary'].values

# =============================================================================
# Dividing the dataset into Training and testing set. Training set is used to train the model and 
# testing set is used to test the model's prediction
# =============================================================================

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#Here, the best fit slope i.e the parameters B0 and B1 (m and b) will get calculated.
def best_fit_slope(x,y):
    m= ((mean(x)*mean(y))-mean(x*y))/((mean(x)*mean(x))-mean(x*x))
    b= (mean(y)-m*mean(x))
    return m,b

m,b=best_fit_slope(X_train,y_train)

#Just subsituting the value of latest x input in the parameters calculated earlier. 
def predict(m,x,b):
    y=(m*x)+b
    return y

pred_values=[predict(m,i,b) for i in X_test]


#Visualizing the results.
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, pred_values, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Calculating the r-squared error.
ssr=0
sst=0
for i in range(len(y_test)):
    ssr+=(y_test[i]-pred_values[i])**2
    sst+=(y_test[i]-mean(y_test))**2
r=1-(ssr/sst)

print("R-Squared error is: ",r)


# =============================================================================
# In the code below, I have used the sklearn library to predict the values. The r-squared value of hardcoded
# prediction and the prediction made by the libraries' method is the same. 
# =============================================================================
x_train,x_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3)
regressor=LinearRegression()
x_train=x_train.reshape(-1,1)
Y_train=Y_train.reshape(-1,1)

regressor.fit(x_train,Y_train)
x_test=x_test.reshape(-1,1)
y_pred=regressor.predict(x_test)
y_test=y_test.reshape(-1,1)
plt.scatter(x_test, Y_test, color = 'red')
plt.plot(x_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print("R-Squared error is: ",r2_score(y_test,pred_values))



