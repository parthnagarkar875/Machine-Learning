# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:51:35 2020

@author: Parth
"""


# =============================================================================
# --> Used for clustering the data points based on their traits into different categories. 
# 
# --> Steps are:
#     a) Choose K number of clusters.
#     b) Randomly initialize K cluster centroids.
#     c) Assign each datapoint to the closest centroid. 
#     d) Compute the mean of the datapoints of different clusters and plot the new centroid.
#     e) Reassign each datapoint to the closest centroid based and do this till no reassignment takes place. 
# 
# =============================================================================


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df=pd.read_csv('Mall_Customers.csv')
X=df.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss=list()

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
