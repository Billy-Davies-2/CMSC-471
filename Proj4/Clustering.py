# Clustering.py
# Coded by: William Davies
# Date: 05/16/2016
# Description: This project will take a file and the number of clusters
#   desired from the user, and perform a k-means clustering on the
#   given data.

# imports
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

NUM_CLUSTERS = int(sys.argv[1])

FILE = sys.argv[2]
myFile = open(FILE, 'r')
X = np.loadtxt(FILE)
plt.figure(figsize=(12, 12))

#n_samples = 1500
#random_state = 170
#X, y = make_blobs(n_samples=n_samples, random_state=random_state)
random_state = 20
# Incorrect number of clusters
y_pred = KMeans(n_clusters=NUM_CLUSTERS, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("K MEANS CLUSTERING")

plt.show()
