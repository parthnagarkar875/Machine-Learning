# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 21:43:02 2020

@author: Parth
"""


# =============================================================================
# --> https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833
# 
# --> The training data is divided into K folds ensuring that every fold is used as a testing set 
#     at some iteration.
#     
# --> This approach is used to test the accuracy of the models for different inputs
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


df0=pd.read_csv('Social_Network_Ads.csv')

df2=pd.get_dummies(df0['Gender'])

result = pd.concat([df0,df2],axis=1)
df=result.drop(['User ID','Gender'],axis=1)

y=df.iloc[:,2].values
df0=df.drop(['Purchased'],axis=1)


X=df0.iloc[:,[0,1]].values



X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

cm=confusion_matrix(y_test,y_pred)

precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
fscore=f1_score(y_test,y_pred)
specificity=cm[0,0]/(cm[0,0]+cm[0,1])

print('Precision: {}'.format(precision))
print('Recall/Sensitivity: {}'.format(recall))
print('F1 score: {}'.format(fscore))
print('Specificity: {}'.format(specificity))

#K-fold cross validation part
accuracies=cross_val_score(estimator=classifier, X=X_train, y=y_train,cv=10)
accuracies.mean()
accuracies.std()


parameters=[{'C':[1,10,100,1000],'kernel':['linear']},
            {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7]}]

grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1)

grid_search=grid_search.fit(X_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()





from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
