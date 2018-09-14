#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:28:04 2018

@author: J. Andrew Casey-Clyde
"""

import os

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt


def cross_val(X, y, folds=5):
    """Cross validation evaluator.

    Performs m-fold cross validation (k-fold in standard conventions, but here
    m-fold is used to avoid confusion) on a series of k-Nearest Neighbors
    classifiers, using k=[1, 10] neighbors for each m-fold cross validation
    run.

    Parameters
    ----------
    X : array_like
        The dataset to use for cross validation, with shape (l, n).
    y : array_like
        The labels corresponding to the datapoints in X. The length of y should
        match the first dimension of X. i.e., it should have length l.
    folds : int
        The number of folds to use in m-fold cross validation, i.e. m.

    Returns
    -------
    `numpy.ndarray`
        An array of the average classification error rate (over m folds) for
        each k=[1, 10].

    """
    # pylint: disable=C0103
    kf = KFold(n_splits=folds)
    avgs = np.array([])
    for k in np.arange(1, 11):  # range is [1, 11)
        knn = KNeighborsClassifier(n_neighbors=k)

        errors = np.array([])

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            knn.fit(X_train, y_train)
            errors = np.append(errors, 1 - knn.score(X_test, y_test))

        avgs = np.append(avgs, errors.mean())

    return avgs


def main():
    """The main run function.

    This is the main function that handles the running of all other functions,
    as well as the overall execution of the solution to problem 1 of HW1.

    """
    # pylint: disable=C0103
    data_path = os.path.join(os.path.dirname(__file__),
                             '..', '..', '..', 'data', 'iris.data')
    data = pd.read_csv(data_path, header=None,
                       names=["sepal-length", "sepal-width",
                              "petal-length", "petal-width", "class"])
    y = data.pop("class").values
    X = data.values
    avgs = {"5-fold cross validation": cross_val(X, y, folds=5),
            "10-fold cross validation": cross_val(X, y, folds=10),
            "Leave-one-out cross validation": cross_val(X, y, folds=len(y))}

    plt.figure(figsize=(12, 12))
    for key, marker in zip(avgs, ["b^-", "gv-", "ro-"]):
        plt.plot(range(1, 11), avgs[key], marker, label=key)

    plt.title("$k$NN: Error rate vs. $k$ neighbors")
    plt.xlabel("$k$")
    plt.ylabel("Average Error Rate")
    plt.legend()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'doc',
                             'img', 'p1.pdf')
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    main()
