# Coded by:             William Davies
# Date:                 04/24/2016
# Class:                CMSC 471
# Description:
#   This project uses K-nearest neighbors to
# create an image classification for each of the 5
# classes.

# imports
import pandas as pd
import numpy as np
import pylab as pl
import PIL
from PIL import Image
import os
import base64
import io
from io import BytesIO
import sys
# sklearn machine learning software imports
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier

# Image conversion section:
#setup a standard image size; this will distort some images but will get everything into the same shape
STANDARD_SIZE = (100, 100)
def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = PIL.Image.open(filename)
    if verbose==True:
        print ("changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE)))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = np.array(img)
    return img

def flatten_image(img):
    """
    takes in an (m, n) numpy array and flattens it
    into an array of shape (1, m * n)
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

# string to image
def string_to_img(image_string):
    #we need to decode the image from base64
    image_string = base64.b64decode(image_string)
    #since we're seing this as a JSON string, we use StringIO so it acts like a file
    img = io.BytesIO(image_string)
    img = PIL.Image.open(img)
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return pca.transform(img_wide[0])

# Data for the training set.
# TODO PATH TO YOUR DATA
img_dir1 = "../Training/"
elements = os.listdir(img_dir1)
images1 = [img_dir1 + f for f in elements[:49]]
images2 = [img_dir1 + f for f in elements[50:99]]
images3 = [img_dir1 + f for f in elements[100:149]]
images4 = [img_dir1 + f for f in elements[150:199]]
images5 = [img_dir1 + f for f in elements[200:]]
images = images1[:] + images2[:] + images3[:] + images4[:] + images5[:]
labels = []
for f in images:
    if f in images1:
        labels.append("Smile")
    elif f in images2:
        labels.append("Hat")
    elif f in images3:
        labels.append("Hash")
    elif f in images4:
        labels.append("Heart")
    elif f in images5:
        labels.append("Dollar")
data = []
for image in images:
    img = img_to_matrix(image)
    img = flatten_image(img)
    data.append(img)

data = np.array(data)
# Create arrays for each of the training and test sets:
is_train = np.random.uniform(0, 1, len(data)) <= 0.7
y  = np.where(np.array(labels) == "Smile", 1, 0)
y1 = np.where(np.array(labels) == "Hat", 1, 0)
y2 = np.where(np.array(labels) == "Hash", 1, 0)
y3 = np.where(np.array(labels) == "Heart", 1, 0)
y4 = np.where(np.array(labels) == "Dollar", 1, 0)

train_x, train_y, train_y1, train_y2, train_y3, train_y4 = data[is_train], y[is_train], y1[is_train], y2[is_train], y3[is_train], y4[is_train]
test_x, test_y, test_y1, test_y2, test_y3, test_y4 = data[is_train==False], y[is_train==False], y1[is_train==False], y2[is_train==False], y3[is_train==False], y4[is_train==False]

# KNN classifier for the data:
pca = RandomizedPCA(n_components=5)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)
knn = KNeighborsClassifier()

# Classify the given set.
def classify_set(yref):
    knn.fit(train_x, yref)

# Tabulate the results for each given set, used in testing
def display_results(yval):
    pd.crosstab(yval, knn.predict(test_x), rownames=["Actual"], colnames=["Predicted"])

# Image classification section:
def classify_image(data, set_space):
    preds = knn.predict(data)
    preds1 = np.where(preds==1, set_space, "NotSet")
    pred = preds1[0]
    if pred == set_space:
        print(set_space)
        return True

def classify(img_string):
    img = string_to_img(img_string)
    for i in range (0, 5):
        if i == 0:
            train_set = train_y
            test_set = "Smile"
            test_check = test_y
        elif i == 1:
            train_set = train_y1
            test_set = "Hat"
            test_check = test_y1
        elif i == 2:
            train_set = train_y2
            test_set = "Hash"
            test_check = test_y2
        elif i == 3:
            train_set = train_y3
            test_set = "Heart"
            test_check = test_y3
        elif i == 4:
            train_set = train_y4
            test_set = "Dollar"
            test_check = test_y4
        classify_set(train_set)
        #display_results(test_check)
        pred = classify_image(img, test_set)

def main():
    filename = sys.argv[-1]
    new_image = open(filename, 'rb').read()

    #new_image = open('./Data/01/51.jpg', 'rb').read()

    #we need to make the image JSON serializeable
    new_image = base64.b64encode(new_image)

    classify(new_image)
main()