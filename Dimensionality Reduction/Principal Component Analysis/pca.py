# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:58:31 2020

@author: Parth
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score

df=pd.read_csv('Wine.csv')

X=df.iloc[:,0:13].values
y=df.iloc[:,13].values


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)

# =============================================================================
# Whenever we have to use dimensionality reduction, it is necessary to scale the features. 
# =============================================================================

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# =============================================================================
# Applying PCA
# =============================================================================

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)
explainedvariance=pca.explained_variance_ratio_ #Explains the ratio of variance for each attribute



classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
y_predSir = classifier.predict_proba(X_test)

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
print("Mean of accuracies:",accuracies.mean())
print("STD of accuracies:",accuracies.std())



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
plt.title('Logistic Regression (Training set)')
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
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
