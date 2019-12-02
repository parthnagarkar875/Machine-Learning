# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:59:24 2019

@author: Parth
"""

'''
Random Forrest Regression.
--->Random Forrest Regression comes under a set of learning algorithms called as Ensemble Learning.

--->Ensemble Learning is the process of taking multiple algorithms or the same algorithm multiple 
    times to generate something more powerful than the original algorithm. 
    
--->Steps in Random Forrest Regression are:
    1) Select any random K points from the dataset. 
    2) Create a Decision tree to predict the dependent variable y on those k data points. 
    3) Specify the number of decision trees to be constructed. 
    4) Repeat steps 1&2 once the number of decision trees are specified in Step 3. 
    5) Let all the Decision Trees predict the dependent variable Y and the end result is computed by 
       taking the average of the results of all the Decision Trees. 
       
--->Ensemble algorithms are more stable as any changes that might affect a single tree do not affect
    the end outcome as the average of the outcomes of all trees is taken. 
    

'''
