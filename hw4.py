#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 09:20:18 2018

@author: jacaseyclyde
"""

import os

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def nlc(X, y, k):
    X_train, X_test = X
    y_train, y_test = y
    distances = None
    y_classes = np.unique(y_train)
    for j in y_classes:
        # for the current class, get the k nearest neighbors for each
        # test point, without individual distances
        clf = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        X_cls = X_train[y_train == j]
        clf.fit(X_cls)

        # calculate the centroid of the k nearest neighbors in class j
        # for each test point
        cents = np.mean(X_cls[clf.kneighbors(X_test, return_distance=False)],
                        axis=1)

        if distances is not None:
            # calculate the distance between each point and it's local
            # centroid for the current class. This is not necessarily
            # the average of the distance to each point comprising the
            # centroid
            distance = np.array([np.diagonal(pairwise_distances(X_test,
                                                                cents))]).T
            distances = np.append(distances, distance, axis=1)
        else:
            distances = np.array([np.diagonal(pairwise_distances(X_test,
                                                                 cents))]).T

    conf = confusion_matrix(y_test, y_classes[np.argmin(distances, axis=1)])
    return np.sum(np.diagonal(conf)) / np.sum(conf)


def main():
    # read in usps data
    # training data
    data = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    'data', 'usps', 'zip.train'),
                       header=None, delimiter=' ').iloc[:, :-1]
    y_train = data.pop(0).values
    X_train = data.values

    # test data
    data = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    'data', 'usps', 'zip.test'),
                       header=None, delimiter=' ')
    y_test = data.pop(0).values
    X_test = data.values

    # PROBLEM 1
    ####################################################################
    # apply pca with 95% variance
    pca = PCA(n_components=.95)
    pca.fit(X_train)

    X_train_95 = pca.transform(X_train)
    X_test_95 = pca.transform(X_test)

    # apply pca with 98% variance
    pca = PCA(n_components=.98)
    pca.fit(X_train)

    X_train_98 = pca.transform(X_train)
    X_test_98 = pca.transform(X_test)

    # apply LDA classification
    lda = LDA()
    lda.fit(X_train_95, y_train)
    err_rate = 1 - lda.score(X_test_95, y_test)
    print(err_rate)

if __name__ == '__main__':
    font = {'size': 24}
    mpl.rc('font', **font)
    directory = os.path.join(os.path.dirname(__file__), 'hw3')
    if not os.path.exists(directory):
        os.makedirs(directory)
#    plt.xkcd()
    main()
