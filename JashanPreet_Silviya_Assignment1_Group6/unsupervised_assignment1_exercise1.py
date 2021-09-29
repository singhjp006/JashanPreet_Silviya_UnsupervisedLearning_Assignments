#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 23:00:06 2021

@author: jashan
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# 1. Retrieve and load the mnist_784 dataset of 70,000 instances.
from sklearn.datasets import fetch_openml
MNIST_jashan = fetch_openml('mnist_784', version=1)

# Listing the keys
print("Keys in MNIST Dataset:\n", MNIST_jashan.keys())

# Assigning the data to a ndarray named X_firstname 
# Assigning the target to a variable named y_firstname
X_jashan, y_jashan = MNIST_jashan["data"].to_numpy(), MNIST_jashan["target"].to_numpy()

#Printing the types of X_firstname and y_firstname.
print("Data types of data and target respectively are:\n", type(X_jashan), " and ", type(y_jashan))

# Printing the shape of X_firstname and y_firstname.
print("Shape of data and target respectively are:\n", X_jashan.shape, " and ", y_jashan.shape)

# printing digits
#Defining a function to plot a group of images 10 per row
def plot_digits(instances, images_per_row=10, **options):
    size = 28
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

# Calling the function to print the first 100 images
plt.figure(figsize=(9,9))
example_images = X_jashan[:100]
plot_digits(example_images, images_per_row=10)
plt.show()


# 3. Use PCA to retrieve the 1th and 2nd principal component and output their explained variance ratio.

# scaling the data
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X_jashan)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_scaled)

print(pca.components_)

print("Explained variance ratio:", pca.explained_variance_)


# 4. Plot the projections of the 1th and 2nd principal component onto a 1D hyperplane.
projected = pca.fit_transform(X_scaled)


plt.scatter(projected[:, 0], projected[:, 1], edgecolor='none', alpha=0.5)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();


#5. Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions.

from sklearn.decomposition import IncrementalPCA
incremental_pca = IncrementalPCA(n_components=154, batch_size=160)

# incremental_pca.fit(X_jashan)

X_pca = incremental_pca.fit_transform(X_scaled)

print("original shape:   ", X_jashan.shape)
print("transformed shape:", X_pca.shape)

print(incremental_pca.explained_variance_ratio_)


print("Variance explained by all the components: ", 
      sum(incremental_pca.explained_variance_ratio_ * 100))


#Display the original and compressed digits from (5). [5 points]

plt.scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='none', alpha=0.5)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();



def plot_digits_x_pca(instances, images_per_row=10, **options):
    size = 12
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(14,11) for instance in instances]
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




# Calling the function to print the first 100 images
plt.figure(figsize=(9,9))
example_images = X_pca[:100]
plot_digits_x_pca(example_images, images_per_row=10)
plt.show()












































