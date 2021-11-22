#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 18:28:35 2021

@author: silviyavelani
         jashanpreetsingh
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import mixture
import numpy as np
import itertools
from scipy import linalg
import matplotlib as mpl

data, target = datasets.fetch_olivetti_faces(shuffle = True,
                                                     return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                    test_size = 0.20,
                                                    stratify = target);

'''1. Use PCA preserving 99% of the variance to reduce the dataset’s dimensionality.'''
pca = PCA(0.99)
pca.fit(X_train)
pca_X_train = pca.transform(X_train)

print(pca.explained_variance_ratio_.sum())


'''2. Determine the most suitable covariance_type for the dataset. '''
lowest_bic = np.infty
bic = []
n_components_range = range(1, 10)
cv_types = ["diag", "tied", "spherical", "full"]
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type=cv_type
        )
        gmm.fit(pca_X_train)
        bic.append(gmm.bic(pca_X_train))


bic = np.array(bic)
color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + 0.2 * (i - 2)
    bars.append(
        plt.bar(
            xpos,
            bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
            width=0.2,
            color=color,
        )
    )
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
plt.title("BIC score per model")
xpos = (
    np.mod(bic.argmin(), len(n_components_range))
    + 0.65
    + 0.2 * np.floor(bic.argmin() / len(n_components_range))
)
plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
spl.set_xlabel("Number of components")
spl.legend([b[0] for b in bars], cv_types)


best_gmm = mixture.GaussianMixture(
            n_components=3, covariance_type="full"
        )
best_gmm.fit(pca_X_train)
clf = best_gmm


'''3. Determine the minimum number of clusters that best represent the dataset using either AIC or BIC.'''
# Plot the winner
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(pca_X_train)
for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(pca_X_train[Y_ == i, 0], pca_X_train[Y_ == i, 1], 0.8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)

plt.xticks(())
plt.yticks(())
plt.title(
    f"Selected GMM: {best_gmm.covariance_type} model, "
    f"{best_gmm.n_components} components"
)
plt.subplots_adjust(hspace=0.35, bottom=0.02)
plt.show()

# Plot the winner
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(pca_X_train)
for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(pca_X_train[Y_ == i, 0], pca_X_train[Y_ == i, 1], 0.8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    
    
'''4. Plot the results from (2) and (3). '''
plt.xticks(())
plt.yticks(())
plt.title(
    f"Selected GMM: {best_gmm.covariance_type} model, "
    f"{best_gmm.n_components} components"
)
plt.subplots_adjust(hspace=0.35, bottom=0.02)
plt.show()


'''5. Output the hard clustering for each instance'''
hard = best_gmm.predict(pca_X_train)


'''6. Output the soft clustering for each instance.'''
soft = best_gmm.predict_proba(pca_X_train)


'''7. Use the model to generate some new faces (using the sample() method), 
and visualize them (use the inverse_transform() method to transform the data back 
                    to its original space based on the PCA method used).'''
# we then tried plotting faces
def plot_faces(instances, images_per_row=5, **options):
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
x_new, y_new = best_gmm.sample(25)
x_inv = pca.inverse_transform(x_new)

plot_faces(x_inv, images_per_row=5)
plt.show()

'''8. Modify some images (e.g., rotate, flip, darken).'''
x_inv_rotate = []

for i in range(0,25):
    x_inv_rotate.append(x_inv[i].reshape([64,64]).T)

plot_faces(x_inv_rotate, images_per_row=5)
plt.show()


x_inv_flip = []

for i in range(0,25):
    x_inv_flip.append(x_inv[i][::-1].reshape([64,64]))
  
plot_faces(x_inv_flip, images_per_row=5)
plt.show()

x_inv_dark = x_inv
x_inv_dark[:,1:-1] *= 0.3
x_inv_dark = x_inv_dark.reshape(-1, 64*64)
  
plot_faces(x_inv_dark, images_per_row=5)
plt.show()


'''9. Determine if the model can detect the anomalies produced in (8) by comparing 
the output of the score_samples() method for normal images and for anomalies).'''
score_ann = best_gmm.score_samples(pca_X_train)

#Counted threshold at percentile 4%
#All the points below the percentile 4%, will be considered as anomalies

thrsld = np.percentile(score_ann,4)
annomalies = score_ann[score_ann<thrsld]

#Calculating same for our original dataset
score = best_gmm.score_samples(pca_X_train)

thrsld1 = np.percentile(score,4)
annomalies1 = score[score<thrsld1]

print(f'Threasold: {thrsld1}')
print(f'Annomalies: {annomalies1}')
