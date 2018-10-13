#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:21:20 2018

@author: jacaseyclyde
"""
import os
import sys
import itertools

from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker


def pca_lda_plot(X, y, components=0.95,
                 title="95% PCA + LDA", out='hw3/pca_lda.pdf'):
    pca = PCA(n_components=components)
    X_pca = pca.fit_transform(X)

    lda = LDA()
    X_lda = lda.fit_transform(X_pca, y)

    marker = itertools.cycle(('.', '+', 'v', '^', 's'))

    if X_lda.shape[1] == 1:
        plt.figure(figsize=(12, 2))
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('data', 0))

        for cls in np.unique(y):
            X_cls = X_lda[y == cls]
            plt.scatter(X_cls[:, 0], [0] * len(X_cls[:, 0]),
                        label=cls, marker=next(marker))

    else:
        plt.figure(figsize=(12, 12))
        for cls in np.unique(y):
            X_cls = X_lda[y == cls]
            plt.scatter(X_cls[:, 0], X_cls[:, 1],
                        label=cls, marker=next(marker))

    lgd = plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.title(title)

    save_path = os.path.join(os.path.dirname(__file__), out)
    plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def pca_lda(X, y, lda_flag=True, nlc_flag=False):
    kmax = 10
    X_train, X_test = X
    y_train, y_test = y

    pca = PCA(n_components=.95)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    if lda_flag:
        lda = LDA()
        lda.fit(X_train, y_train)
        X_train = lda.transform(X_train)
        X_test = lda.transform(X_test)

        desc = "PCA + LDA + kNN"
    else:
        desc = "PCA + kNN"

    errors = np.array([])
    if nlc_flag:
        for k in tqdm(np.arange(1, kmax + 1), desc=desc, file=sys.stdout):
            errors = np.append(errors, 1 - nlc((X_train, X_test),
                                               (y_train, y_test), k))
    else:
        for k in tqdm(np.arange(1, kmax + 1), desc=desc, file=sys.stdout):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)

            errors = np.append(errors, 1 - knn.score(X_test, y_test))

    return errors


def nlc(X, y, k):
    """Nearest Local Centroid classifier.

    This classifier uses the nearest local centroid in the training
    data to classify points in the testing data.

    Parameters
    ----------
    X : (array_like, array_like)
        Feature columns of data to use for training/testing,
        respectivley.
    y : (array_liuke, array_like)
        Classes for training/testing data, respectivley.
    k : int
        Number of neighbors to use for each centroid.

    Returns
    -------
    array_like
        The predicted classes.

    """
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


def plot_results(errors, title="Error rate vs. $k$ Neighbors",
                 out='errors.pdf'):
    plt.figure(figsize=(12, 12))
    marker = itertools.cycle(('.', '+', 'v', '^', 's'))
    for key in errors:
        plt.plot(range(1, 11), errors[key], label=key, marker=next(marker))

    plt.legend()

    plt.title(title)
    plt.xlabel("$k$")
    plt.ylabel("Error Rate")
    plt.grid()
    save_path = os.path.join(os.path.dirname(__file__), out)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def min_vals(vals, label="Values"):
    min_val = np.inf
    min_key = ""
    min_k = 0
    for k in vals.keys():
        val = vals[k]
        if isinstance(val, tuple):
            val = np.sum(val)
        if np.min(val) < min_val:
            min_val = np.min(val)
            min_key = k
            min_k = np.where(val == min_val)[0][0]

    print("{0}: min val = {1}, label = {2}, k = {3}".format(label, min_val,
                                                            min_key,
                                                            min_k + 1))


def main():
    # Problem 1
    # load iris data
    iris_data = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                         'data', 'iris.data'))
    y = iris_data.pop("species").values

    pca = PCA(n_components=2)
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(pca.fit_transform(iris_data.values), y)

    pca_lda_plot(X_lda, y, components=2,
                 title="2 Component PCA + LDA", out='hw3/pca_lda_2c.pdf')

    # Problem 3
    data = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    'data', 'usps', 'zip.train'),
                       header=None, delimiter=' ').iloc[:, :-1]
    y_train = data.pop(0).values
    X_train = data.values

    # 3a
    pca_lda_plot(np.append(X_train[y_train == 0],
                           X_train[y_train == 1], axis=0),
                 np.append(y_train[y_train == 0],
                           y_train[y_train == 1], axis=0),
                 title="95% PCA + LDA, USPS 0, 1",
                 out='hw3/pca_lda_01.pdf')

    # 3b
    pca_lda_plot(np.append(X_train[y_train == 4],
                           X_train[y_train == 9], axis=0),
                 np.append(y_train[y_train == 4],
                           y_train[y_train == 9], axis=0),
                 title="95% PCA + LDA, USPS 4, 9",
                 out='hw3/pca_lda_49.pdf')

    # 3c
    pca_lda_plot(np.append(np.append(X_train[y_train == 1],
                                     X_train[y_train == 2], axis=0),
                           X_train[y_train == 3], axis=0),
                 np.append(np.append(y_train[y_train == 1],
                                     y_train[y_train == 2], axis=0),
                           y_train[y_train == 3], axis=0),
                 title="95% PCA + LDA, USPS 1, 2, 3",
                 out='hw3/pca_lda_123.pdf')

    # 3d
    pca_lda_plot(np.append(np.append(X_train[y_train == 3],
                                     X_train[y_train == 5], axis=0),
                           X_train[y_train == 8], axis=0),
                 np.append(np.append(y_train[y_train == 3],
                                     y_train[y_train == 5], axis=0),
                           y_train[y_train == 8], axis=0),
                 title="95% PCA + LDA, USPS 3, 5, 8",
                 out='hw3/pca_lda_358.pdf')

    # Problem 4
    data = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    'data', 'usps', 'zip.test'),
                       header=None, delimiter=' ')
    y_test = data.pop(0).values
    X_test = data.values

    print("--- kNN ---")
    knn_errors = {"PCA + LDA": pca_lda((X_train, X_test), (y_train, y_test),
                                       lda_flag=True),
                  "PCA": pca_lda((X_train, X_test), (y_train, y_test),
                                 lda_flag=False)
                  }

    plot_results(knn_errors, title="$k$NN: Error rate vs. $k$ Neighbors",
                 out='hw3/knn_errors.pdf')

    min_vals(knn_errors, label="kNN Errors")

    # Problem 5
    print("--- NLC ---")
    nlc_errors = {"PCA + LDA": pca_lda((X_train, X_test), (y_train, y_test),
                                       lda_flag=True, nlc_flag=True),
                  "PCA": pca_lda((X_train, X_test), (y_train, y_test),
                                 lda_flag=False, nlc_flag=True)
                  }

    plot_results(nlc_errors, title="NLC: Error rate vs. $k$ Neighbors",
                 out='hw3/nlc_errors.pdf')

    min_vals(nlc_errors, label="NLC Errors")


if __name__ == '__main__':
    font = {'size': 24}
    mpl.rc('font', **font)
    directory = os.path.join(os.path.dirname(__file__), 'hw3')
    if not os.path.exists(directory):
        os.makedirs(directory)
#    plt.xkcd()
    main()
