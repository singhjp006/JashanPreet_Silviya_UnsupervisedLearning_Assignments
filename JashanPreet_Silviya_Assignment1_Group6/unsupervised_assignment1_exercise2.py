#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:35:02 2021

@author: silviyavelani
"""

import numpy as np
import matplotlib.pyplot as plt


#----------------------------------------------------------------------
#1. Generate Swiss roll dataset. [5 points]

from sklearn import datasets
X_Silviya, y_Silviya = datasets.make_swiss_roll(n_samples=1000, noise=0.0)



#----------------------------------------------------------------------
#2. Plot the resulting generated Swiss roll dataset. [2 points]

fig = plt.figure()

aX_Silviya = fig.add_subplot(211, projection='3d')
aX_Silviya.scatter(X_Silviya[:, 0], X_Silviya[:, 1], X_Silviya[:, 2], c=y_Silviya, cmap=plt.cm.Spectral)

aX_Silviya.set_title("Swiss Roll Dataset Plot")
plt.show()



#----------------------------------------------------------------------
#3. Use Kernel PCA (kPCA) with linear kernel (2 points), a RBF kernel (2 points), 
#and a sigmoid kernel (2 points). [6 points]

from sklearn.decomposition import KernelPCA

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Silviya = sc.fit_transform(X_Silviya)


transformer_linear = KernelPCA(n_components=2, kernel='linear')
X_Silviya_transformed_linear = transformer_linear.fit_transform(X_Silviya)
print(f'Shape of X_Silviya after linear transformer {X_Silviya_transformed_linear.shape}')


transformer_rbf = KernelPCA(n_components=2, kernel='rbf')
X_Silviya_transformed_rbf = transformer_rbf.fit_transform(X_Silviya)
print(f'Shape of X_Silviya after rbf transformer {X_Silviya_transformed_rbf.shape}')


transformer_sigmoid = KernelPCA(n_components=2, kernel='sigmoid')
X_Silviya_transformed_sigmoid = transformer_sigmoid.fit_transform(X_Silviya)
print(f'Shape of X_Silviya after sigmoid transformer {X_Silviya_transformed_sigmoid.shape}')

"""The kernel functions are used to map the original dataset (linear/nonlinear ) into a 
higher dimensional space with view to making it linear dataset. Usually linear and polynomial 
kernels are less time consuming and provides less accuracy than the rbf or Gaussian kernels."""

#----------------------------------------------------------------------
#4. Plot the kPCA results of applying the linear kernel (2 points), a RBF kernel (2 points), 
# and a sigmoid kernel (2 points) from (3). Explain and compare the results [6 points]

fig = plt.figure()

linear_X_Silviya = fig.add_subplot(211, projection='3d')
linear_X_Silviya.scatter(X_Silviya_transformed_linear[:, 0], X_Silviya_transformed_linear[:, 1], c=y_Silviya, cmap=plt.cm.Spectral)

linear_X_Silviya.set_title("X_Silviya_transformed_linear")
plt.show()

###############################
fig = plt.figure()

rbf_X_Silviya = fig.add_subplot(211, projection='3d')
rbf_X_Silviya.scatter(X_Silviya_transformed_rbf[:, 0], X_Silviya_transformed_rbf[:, 1], c=y_Silviya, cmap=plt.cm.Spectral)

rbf_X_Silviya.set_title("X_Silviya_transformed_rbf")
plt.show()

##############################
fig = plt.figure()

sigmoid_X_Silviya = fig.add_subplot(211, projection='3d')
sigmoid_X_Silviya.scatter(X_Silviya_transformed_sigmoid[:, 0], X_Silviya_transformed_sigmoid[:, 1], c=y_Silviya, cmap=plt.cm.Spectral)

sigmoid_X_Silviya.set_title("X_Silviya_transformed_sigmoid")
plt.show()



#----------------------------------------------------------------------
#5. Using kPCA and a kernel of your choice, apply Logistic Regression for classification. 
#Use GridSearchCV to find the best kernel and gamma value for kPCA in order to get 
#the best classification accuracy at the end of the pipeline. Print out best parameters found by GridSearchCV. [14 points]
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline(
    [("kpca", KernelPCA(n_components=2)),
     ("log_reg", LogisticRegression())])

param_grid = [{"kpca__gamma": np.linspace(0.03, 0.05, 10),
               "kpca__kernel": ["sigmoid", "rbf", "linear"]}]

grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_Silviya.astype(np.int), y_Silviya.astype(np.int))



#----------------------------------------------------------------------
#6. Plot the results from using GridSearchCV in (5). [2 points]
print(f'Best Param: {grid_search.best_params_}')
print(f'Best Estimator: {grid_search.best_estimator_}')
print(f'Best Score: {grid_search.best_score_}')
print(f'CV Result: {grid_search.cv_results_}')


