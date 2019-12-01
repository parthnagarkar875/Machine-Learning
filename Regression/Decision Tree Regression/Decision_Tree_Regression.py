# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:00:01 2019

@author: Parth
"""
#https://www.saedsayad.com/decision_tree.htm
'''
--->Decision Tree algorithm constructs a tree like structure to divide the labels based 
    on the conditional values. 

--->The concept of Entropy and Information gain is used in Decision Trees. 

--->Entropy is a measure of the IMPURITY present in the data. We initially calculate the entropy of 
    the target variable and then subtract the entropy from the entropy of eah feature. 
    The subtracted value then obtained is called as Information Gain. 

--->The feature with the highest information gain is placed is placed as the child of the root node. 
'''

