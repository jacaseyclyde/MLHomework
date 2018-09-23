#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 12:19:26 2018

@author: jacaseyclyde
"""

import os

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


import matplotlib.pyplot as plt

from tqdm import tqdm


def inverse_dist(distances):
    """Inverse distance weighting function.

    Weighting function that uses the inverse distance to weight each neighbor
    in kNN classification.

    Parameters
    ----------
    distances : array_like
        The distance each neighbor is from the test point

    Returns
    -------
    array_like
        The weight to use for each neighbor.

    """
    return 1. / distances


def square_inverse_dist(distances):
    """Square inverse distance weighting function.

    Weighting function that uses the square inverse distance to weight each
    neighbor in kNN classification.

    Parameters
    ----------
    distances : array_like
        The distance each neighbor is from the test point

    Returns
    -------
    array_like
        The weight to use for each neighbor.

    """
    return 1. / (distances**2)


def linear_dist(distances):
    """Linear distance weighting function.

    Weighting function that uses a linear function to weight each neighbor in
    kNN classification.

    Parameters
    ----------
    distances : array_like
        The distance each neighbor is from the test point

    Returns
    -------
    array_like
        The weight to use for each neighbor.

    """
    return (np.max(distances) - distances) / \
           (np.max(distances) - np.min(distances))


def nlc(X, y, k, metric='euclidean'):
    """Nearest Local Centroid classifier.

    This classifier uses the nearest local centroid in the training data to
    classify points in the testing data.

    """
    X_train, X_test = X
    y_train, y_test = y
    distances = None
    for j in np.unique(y_train):
        # for the current class, get the k nearest neighbors for each test
        # point, without individual distances
        clf = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        X_cls = X_train[y_train == j]
        clf.fit(X_cls)
        neighbors = clf.kneighbors(X_test, return_distance=False)

        # calculate the centroid of the k nearest neighbors in class j for
        # each test point
        cents = np.mean(X_cls[neighbors], axis=1)

        if distances is not None:
            # calculate the distance between each point and it's local centroid
            # for the current class. This is not necessarily the average of the
            # distance to each point comprising the centroid
            distance = np.array([np.diagonal(pairwise_distances(X_test,
                                                                cents))]).T
            distances = np.append(distances, distance, axis=1)
        else:
            distances = np.array([np.diagonal(pairwise_distances(X_test,
                                                                 cents))]).T

    return np.argmin(distances, axis=1)

def k_iter(X, y, key, classifier=None, folds=None,
           metric='euclidean', weights='uniform'):
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
    folds : int, optional (default = None)
        The number of folds to use in m-fold cross validation. If `None`,
        cross-validation will not be performed.
    metric : string, optional (default = 'euclidean')
        The distance metric to use.
    weights : string or callable, optional (default = 'uniform')
        The weighting function to use.

    Returns
    -------
    `numpy.ndarray`
        An array of the classification error rate for each k=[1:10], or the
        average rate if `folds` is not `None`.

    """
    # pylint: disable=C0103

    errors = np.array([])
    for k in tqdm(np.arange(1, 11), desc=key):  # range is [1, 11)
        if classifier is not None:
            y_train, y_test = y

            y_pred = classifier(X, y, k)

            conf = confusion_matrix(y_test, y_pred)
            error = 1 - np.sum(np.diagonal(conf)) / np.sum(conf)
        else:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric,
                                       weights=weights, algorithm='auto',
                                       n_jobs=-1)
            if folds is not None:
                kf = KFold(n_splits=folds)
                errs = np.array([])
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    knn.fit(X_train, y_train)
                    y_pred = knn.predict(X_test)

                    conf = confusion_matrix(y_test, y_pred)
                    err = 1 - np.sum(np.diagonal(conf)) / np.sum(conf)
                    errs = np.append(errs, err)

                error = errs.mean()
            else:
                X_train, X_test = X
                y_train, y_test = y

                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

                conf = confusion_matrix(y_test, y_pred)
                error = 1 - np.sum(np.diagonal(conf)) / np.sum(conf)

        errors = np.append(errors, error)

    return errors


def plot_results(errors, title="$k$NN: Error rate vs. $k$ neighbors",
                 out='out.pdf'):
    plt.figure(figsize=(12, 12))
    if type(errors) is dict:
        for key, marker in zip(errors, ['b^-', 'gv-', 'ro-']):
            plt.plot(range(1, 11), errors[key], marker, label=key)
    else:
        plt.plot(range(1, 11), errors, 'b^-')

    plt.title(title)
    plt.xlabel("$k$")
    plt.ylabel("Average Error Rate")
    plt.legend()
    save_path = os.path.join(os.path.dirname(__file__), out)
    plt.savefig(save_path)
    plt.show()


def main():
    """The main run function.

    This is the main function that handles the running of all other functions,
    as well as the overall execution of the solution to problem 1 of HW1.

    """
    # pylint: disable=C0103
    # Problem 1
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'iris.data')
    data = pd.read_csv(data_path)
    y = data.pop("species").values
    X = data.values

    keys = ["5-fold cross validation", "10-fold cross validation",
            "Leave-one-out cross validation"]
    errors = {keys[0]: k_iter(X, y, keys[0], folds=5),
              keys[1]: k_iter(X, y, keys[1], folds=10),
              keys[2]: k_iter(X, y, keys[2], folds=len(y))}

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

    keys = ["Cosine", "Euclidean", "City-Block"]
    errors = {keys[0]: k_iter(X, y, keys[0], metric='cosine'),
              keys[1]: k_iter(X, y, keys[1], metric='euclidean'),
              keys[2]: k_iter(X, y, keys[2], metric='manhattan')
              }

    title = "$k$NN Distance Metrics: Error rate vs. $k$ neighbors"
    plot_results(errors, title=title, out='p2.pdf')

    # Problem 3
    keys = ["Inverse", "Square Inverse", "Linear"]
    errors = {keys[0]: k_iter(X, y, keys[0], weights=inverse_dist),
              keys[1]: k_iter(X, y, keys[1], weights=square_inverse_dist),
              keys[2]: k_iter(X, y, keys[2], weights=linear_dist)}
    print(type(errors))

    title = "k$NN Weighting Functions: Error rate vs. $k$ neighbors"
    plot_results(errors, title=title, out='p3.pdf')

    # Problem 4
    errors = k_iter(X, y, "NLC", classifier=nlc)

    title = "k$NN Nearest Local Centroid: Error rate vs. $k$ neighbors"
    plot_results(errors, title=title, out='p4.pdf')


if __name__ == "__main__":
    main()
