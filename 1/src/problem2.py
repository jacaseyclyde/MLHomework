#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:28:04 2018

@author: J. Andrew Casey-Clyde
"""

import os

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import KNeighborsClassifier

from tqdm import tqdm

import matplotlib.pyplot as plt


def _cosine_dist(X, Y):
    """Wrapper for cosine distance metric.

    This function is just a wrapper ofr sklearns built in cosine_distance
    metric. This is necessary due to the shapes of each entry, which need to be
    reshaped for the distance metric to handle them properly.

    Parameters
    ----------
    X : array_like
        The first datapoint.
    Y : array_like
        The second datapoint.

    Returns
    -------
    float
        The cosine distance between X and Y.

    """
    cos_dist = cosine_distances(X.reshape(1, -1), Y.reshape(1, -1))
    return cos_dist


def dist(train, test, metric='euclidean', metric_params=None):
    """Distance metric evaluator.

    Performs k Nearest Neighbors classification for k=[1, 10], with a user
    specified distance metric.

    Parameters
    ----------
    X : array_like
        The dataset to use for cross validation, with shape (l, n).
    y : array_like
        The labels corresponding to the datapoints in X. The length of y should
        match the first dimension of X. i.e., it should have length l.
    metric : string of callable
        The distance metric to use.

    Returns
    -------
    `numpy.ndarray`
        An array of classificaiton error rate for each k=[1, 10].

    """
    # pylint: disable=C0103
    X_train, y_train = train
    X_test, y_test = test

    errors = np.array([])
    for k in tqdm(np.arange(1, 11), desc=metric):  # range is [1, 11)
        if metric_params is not None:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric,
                                       metric_params=metric_params)
        else:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

        knn.fit(X_train, y_train)
        errors = np.append(errors, 1 - knn.score(X_test, y_test))

    return errors


def main():
    """The main run function.

    This is the main function that handles the running of all other functions,
    as well as the overall execution of the solution to problem 2 of HW1.

    """
    # pylint: disable=C0103
    train_path = os.path.join(os.path.dirname(__file__),
                              '..', '..', '..', 'data', 'zip.train')
    data = pd.read_csv(train_path, header=None, delimiter=' ').iloc[:, :-1]
    y_train = data.pop(0).values
    X_train = data.values

    test_path = os.path.join(os.path.dirname(__file__),
                             '..', '..', '..', 'data', 'zip.test')
    data = pd.read_csv(test_path, header=None, delimiter=' ')
    y_test = data.pop(0).values
    X_test = data.values

    avgs = {#"Euclidean": dist((X_train, y_train), (X_test, y_test),
            #                  metric='euclidean'),
            #"City-block": dist((X_train, y_train), (X_test, y_test),
            #                      metric='manhattan'),
            "Cosine": dist((X_train, y_train), (X_test, y_test),
                           metric='pyfunc',
                           metric_params={'func': _cosine_dist})}

    plt.figure(figsize=(12, 12))
    for key, marker in zip(avgs, ['b^-', 'gv-', 'ro-']):
        plt.plot(range(1, 11), avgs[key], marker, label=key)

    plt.title("$k$NN: Error rate vs. $k$ neighbors")
    plt.xlabel("$k$")
    plt.ylabel("Average Error Rate")
    plt.legend()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'doc',
                             'img', 'p2.pdf')
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    main()
