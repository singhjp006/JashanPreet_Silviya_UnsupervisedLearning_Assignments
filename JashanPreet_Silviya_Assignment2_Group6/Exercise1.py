#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 19:46:09 2021

@author: jashan
@author: silviyavelani
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import pandas as pd 




"""1. Retrieve and load the Olivetti faces dataset [5 points]"""
data, target = datasets.fetch_olivetti_faces(shuffle = True,
                                                     return_X_y = True)



"""2. Split the training set, a validation set, and a test set using stratified sampling 
to ensure that there are the same number of images per person in each set. Provide your rationale for the split ratio [10 points]"""
from sklearn.model_selection import train_test_split

X_train, X_test_validate, y_train, y_test_validate = train_test_split(data, target,
                                                    test_size = 0.40,
                                                    stratify = target);

X_test, X_validate, y_test, y_validate = train_test_split(X_test_validate, y_test_validate,
                                                    test_size = 0.50,
                                                    stratify = y_test_validate);

'''We decided to take split ratio of 0.6 for train, 0.2 for validate, and 0.2 for test because
        1. If we set 0.7 for train and 0.15 for validate and test respectively 
           then it is not possible to set same number of images per person in each set  
           
        2. If we set 0.8 for train and 0.1 for validate and test respectively then
           it is not possible to set cv for cross validate on validation data as there are only
           1 person per class
'''

# checking if there are the same number of images per person in each set
unique, counts = np.unique(y_train, return_counts=True)  
frequencies = np.asarray((unique, counts)).T
print("Frequencies of each person in y_train:\n", frequencies)

unique, counts = np.unique(y_test, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print("Frequencies of each person in y_test:\n", frequencies)

unique, counts = np.unique(y_validate, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print("Frequencies of each person in y_validate:\n", frequencies)

# we then tried plotting faces
def plot_faces(instances, images_per_row=10, **options):
    size = 64
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

# Calling the function to print the first 100 faces
plt.figure(figsize=(20,20))
example_images = data[:100]
plot_faces(example_images, images_per_row=10)
plt.show()



"""3. Using k-fold cross validation, train a classifier to predict which person is represented 
in each picture, and evaluate it on the validation set. [30 points]"""


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

LR_clf = LogisticRegression(multi_class="auto", solver="lbfgs", C = 1,
                            tol=1e-1, max_iter = 500)

scores = cross_val_score(LR_clf, X_validate, y_validate, cv=2, scoring="accuracy")
print("Scores on k-fold cross validation: ", scores)
print("Mean of Scores on k-fold cross validation: ", scores.mean())

LR_clf.fit(X_train, y_train)

print("Score of Logistic Regression classifier by using solver lbfgs: \n",
      LR_clf.score(X_validate, y_validate))

y_pred = LR_clf.predict(X_test)
print(accuracy_score(y_pred, y_test))




"""4. Use K-Means to reduce the dimensionality of the set. Provide your rationale for the 
similarity measure used to perform the clustering. Use the silhouette score approach to choose 
the number of clusters. [25 points]"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


#Scaled data using standardscaler and added it back in 'data' variable
scaler = StandardScaler()
scaler.fit(data)
X_scale = scaler.transform(data)
data = pd.DataFrame(X_scale)

from yellowbrick.cluster import KElbowVisualizer
model = KMeans()

# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30),metric='silhouette', timings= True)
visualizer.fit(data)        # Fit the data to the visualizer
visualizer.show() 


data_scale = data.copy()
kmeans_scale = KMeans(n_clusters=2, n_init=100, max_iter=1000, init='k-means++', random_state=42).fit(data_scale)
print('KMeans Scaled Silhouette Score: {}'.format(silhouette_score(data_scale, kmeans_scale.labels_, metric='euclidean')))

labels_scale = kmeans_scale.labels_
clusters_scale = pd.concat([pd.DataFrame(data_scale), pd.DataFrame({'cluster_scaled':labels_scale})], axis=1)

range_n_clusters = [2, 40, 60, 100, 125, 150]

'''After running this graph we decided to go with number of clusters as 2'''
# for n_clusters in range_n_clusters:
#     # Create a subplot with 1 row and 2 columns
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(18, 7)

#     # The 1st subplot is the silhouette plot
#     # The silhouette coefficient can range from -1, 1 but in this example all
#     # lie within [-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     # The (n_clusters+1)*10 is for inserting blank space between silhouette
#     # plots of individual clusters, to demarcate them clearly.
#     ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(data)

#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(data, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)

#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(data, cluster_labels)

#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]

#         ith_cluster_silhouette_values.sort()

#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i

#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)

#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples

#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")

#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

#     # 2nd Plot showing the actual clusters formed
#     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#     ax2.scatter(data[:, 0], data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
#                 c=colors, edgecolor='k')

#     # Labeling the clusters
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#                 c="white", alpha=1, s=200, edgecolor='k')

#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                     s=50, edgecolor='k')

#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")

#     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                   fontsize=14, fontweight='bold')

# plt.show()


C = kmeans_scale.cluster_centers_
C.shape


def extractDistanceFeatures(X,C):
    X_train_r = []
    for i in range(C.shape[0]):
        ci = C[i,:]
        di = np.sum((X-ci)**2,axis=1)
        X_train_r.append(di)
    return np.array(X_train_r).T


X_train_r = extractDistanceFeatures(X_train,C)
X_train_r.shape


kmeans_scale_X_train_r = KMeans(n_clusters=2, n_init=100, max_iter=1000, init='k-means++', random_state=42).fit(X_train_r)
print('KMeans Scaled Silhouette Score: {}'.format(silhouette_score(X_train_r, kmeans_scale_X_train_r.labels_, metric='euclidean')))

'''We can notice that Silhouette Score is now increased to almost 4 times then before'''



"""5. Use the set from (4) to train a classifier as in (3) using k-fold cross validation. [30 points]"""
X_validate_r = extractDistanceFeatures(X_validate,C)

LR_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C = 1,
                            tol=1e-1, max_iter = 1500)

scores = cross_val_score(LR_clf, X_validate_r, y_validate, cv=2, scoring="accuracy")
print("Scores on 3-fold cross validation: ", scores)
print("Mean of Scores on 3-fold cross validation: ", scores.mean())

LR_clf.fit(X_train_r, y_train)

print("Score of Logistic Regression classifier by using solver lbfgs: \n",
      LR_clf.score(X_validate_r, y_validate))

X_test_r = extractDistanceFeatures(X_test,C)

y_pred = LR_clf.predict(X_test_r)
print(accuracy_score(y_pred, y_test))













