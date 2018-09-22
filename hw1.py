#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 12:19:26 2018

@author: jacaseyclyde
"""

import os

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_distances

import matplotlib.pyplot as plt

from tqdm import tqdm


def k_iter(X, y, folds=None, metric='euclidean'):
    """k-neighbors iterator.

    Iterates through [1:k] neighbors for a kNN classifier, allowing for
    comparisons of different kNN setups for differing numbers of neighbors.

    Parameters
    ----------
    X : array_like or (array_like, array_like)
        The dataset(s) to use for training and testing.
    y : array_like or (array_like, array_like)
        The labels corresponding to the datapoints in X. The length of y should
        match the first dimension of X. i.e., it should have length l.
    folds : int, optional
        The number of folds to use in m-fold cross validation.

    Returns
    -------
    `numpy.ndarray`
        An array of the classification error rate for each k=[1:10], or the
        average rate if `folds` is not `None`.

    """
    # pylint: disable=C0103

    errors = np.array([])
    for k in tqdm(np.arange(1, 11), ):  # range is [1, 11)
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric,
                                   algorithm='brute', n_jobs=-1)
        if folds is not None:
            kf = KFold(n_splits=folds)
            scores = np.array([])
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                knn.fit(X_train, y_train)
                scores = np.append(scores, knn.score(X_test, y_test))

            error = 1 - scores.mean()
        else:
            X_train, X_test = X
            y_train, y_test = y

            knn.fit(X_train, y_train)
            error = 1 - knn.score(X_test, y_test)

        errors = np.append(errors, error)

    return errors


def plot_results(errors, title="$k$NN: Error rate vs. $k$ neighbors",
                 out='out.pdf'):
    plt.figure(figsize=(12, 12))
    for key, marker in zip(errors, ["b^-", "gv-", "ro-"]):
        plt.plot(range(1, 11), errors[key], marker, label=key)

    plt.title(title)
    plt.xlabel("$k$")
    plt.ylabel("Average Error Rate")
    plt.legend()
    save_path = os.path.join(os.path.dirname(__file__), out)
    plt.savefig(save_path)


def main():
    """The main run function.

    This is the main function that handles the running of all other functions,
    as well as the overall execution of the solution to problem 1 of HW1.

    """
    # pylint: disable=C0103
    # Problem 1
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'iris.data')
    data = pd.read_csv(data_path, header=None,
                       names=["sepal-length", "sepal-width",
                              "petal-length", "petal-width", "class"])
    y = data.pop("class").values
    X = data.values
    errors = {"5-fold cross validation": k_iter(X, y, folds=5),
              "10-fold cross validation": k_iter(X, y, folds=10),
              "Leave-one-out cross validation": k_iter(X, y, folds=len(y))}

    title = "$k$NN, m-fold Cross Validation: Error rate vs. $k$ neighbors"
    plot_results(errors, title=title, out='p1.pdf')

    # Problem 2
    train_path = os.path.join(os.path.dirname(__file__), 'data', 'zip.train')
    data = pd.read_csv(train_path, header=None, delimiter=' ').iloc[:, :-1]
    y_train = data.pop(0).values
    X_train = data.values

    test_path = os.path.join(os.path.dirname(__file__), 'data', 'zip.test')
    data = pd.read_csv(test_path, header=None, delimiter=' ')
    y_test = data.pop(0).values
    X_test = data.values

    X = (X_train, X_test)
    y = (y_train, y_test)

    errors = {"Cosine": k_iter(X, y, metric=cosine_distances),
              "Euclidean": k_iter(X, y, metric='euclidean'),
              "City-block": k_iter(X, y, metric='manhattan')
              }

    title = "$k$NN Distance Metrics: Error rate vs. $k$ neighbors"
    plot_results(errors, title=title, out='p2.pdf')


if __name__ == "__main__":
    main()
