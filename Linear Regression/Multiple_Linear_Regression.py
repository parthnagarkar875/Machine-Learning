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
    
--->
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('headbrain.csv')























