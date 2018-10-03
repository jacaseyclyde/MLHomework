#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 21:54:48 2018

@author: jacaseyclyde
"""

import os
import sys
import time

from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools

from mnist import MNIST

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


def pca_plot(X, desc, labels=None, out='pca.pdf'):
    """Plots the first two principal components of the given dataset.

    This finds the first two principal directions of the given dataset,
    then plots the projections of the dataset onto these directions
    (i.e., their first two principal components).

    Parameters
    ----------
    X : array_like or sequence of array-like
        The datapoints to be projected with PCA. Each array represents
        a different possible class, and should have shape
        (n_samples, n_features).
    desc : string
        Description of the data. Used for plot title.
    labels : array_like, shape (n_classes), optional (default = `None`)
        The class labels for the datasets. Only used if more than 1
        class given.

    """
    # Combine all classes for computing PCA if more than 1 class given
    multiclass = isinstance(X, tuple)
    if multiclass:
        X_flat = np.concatenate(X, axis=0)
    else:
        X_flat = X

    pca = PCA(n_components=2)
    pca.fit(X_flat)

    plt.figure(figsize=(12, 12))
    if multiclass:
        if labels is None:
            labels = range(len(X))

        marker = itertools.cycle(('.', '+', 'v', '^', 's'))
        for X_y, label in zip(X, labels):
            X_transform = pca.transform(X_y)
            X_transform = X_transform.T
            plt.scatter(X_transform[0], X_transform[1],
                        label=label, marker=next(marker))

        plt.legend()
    else:
        X_transform = pca.transform(X)
        X_transform = X_transform.T
        plt.scatter(X_transform[0], X_transform[1])

    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.title("First Two Principal Components: {0}".format(desc))
    save_path = os.path.join(os.path.dirname(__file__), out)
    plt.savefig(save_path)
    plt.show()


def pca_knn(X, y, var=None):
    kmax = 10
    X_train, X_test = X
    y_train, y_test = y

    if var is not None:
        t0 = time.process_time()
        pca = PCA(var)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        print("--- {0}% variance retained with the first {1} components ---"
              .format(var * 100, pca.n_components_))
        tpca = time.process_time() - t0
    else:
        var = 1.  # just for the tqdm label
        tpca = 0.

    errors = np.array([])
    train_times = np.array([])
    test_times = np.array([])
    for k in tqdm(np.arange(1, kmax + 1), desc="{0}% variance"
                  .format(var * 100), file=sys.stdout):
        t0 = time.process_time()
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        ttrain = time.process_time() - t0

        t0 = time.process_time()
        errors = np.append(errors, 1 - knn.score(X_test, y_test))
        ttest = time.process_time() - t0
        train_times = np.append(train_times, ttrain)
        test_times = np.append(test_times, ttest)

    return errors, (tpca, train_times, test_times)


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
    ttrain = 0.
    ttest = 0.
    for j in y_classes:
        # for the current class, get the k nearest neighbors for each
        # test point, without individual distances
        t0 = time.process_time()
        clf = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        X_cls = X_train[y_train == j]
        clf.fit(X_cls)
        ttrain += time.process_time() - t0
        t0 = time.process_time()
        neighbors = clf.kneighbors(X_test, return_distance=False)

        # calculate the centroid of the k nearest neighbors in class j
        # for each test point
        cents = np.mean(X_cls[neighbors], axis=1)

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
        ttest += time.process_time() - t0

    t0 = time.process_time()
    y_pred = y_classes[np.argmin(distances, axis=1)]
    ttest += time.process_time() - t0
    conf = confusion_matrix(y_test, y_pred)
    return np.sum(np.diagonal(conf)) / np.sum(conf), ttrain, ttest


def pca_nlc(X, y, var=None):
    kmax = 10
    X_train, X_test = X
    y_train, y_test = y

    if var is not None:
        t0 = time.process_time()
        pca = PCA(var)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        print("--- {0}% variance retained with the first {1} components ---"
              .format(var * 100, pca.n_components_))
        tpca = time.process_time() - t0
    else:
        var = 1.  # just for the tqdm label
        tpca = 0.  # no pca means no time

    errors = np.array([])
    train_times = np.array([])
    test_times = np.array([])
    for k in tqdm(np.arange(1, kmax + 1), desc="{0}% variance"
                  .format(var * 100), file=sys.stdout):
        t0 = time.process_time()
        score, ttrain, ttest = nlc((X_train, X_test), (y_train, y_test), k)
        errors = np.append(errors, 1 - score)
        train_times = np.append(train_times, ttrain)
        test_times = np.append(test_times, ttest)

    return errors, (tpca, train_times, test_times)


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
    save_path = os.path.join(os.path.dirname(__file__), out)
    plt.savefig(save_path)
    plt.show()


def plot_times(times, title="Run Time vs. $k$ Neighbors", out='times.pdf'):
    plt.figure(figsize=(12, 12))
    marker = itertools.cycle(('.', '+', 'v', '^', 's', '>', '<', 'o'))
    color = ['green', 'blue', 'orange']
    for i, key in enumerate(times):
        # make plots showing total times, as well as their breakdowns
        tpca, ttrain, ttest = times[key]
        ttot = tpca + ttrain + ttest
        plt.plot(range(1, 11), ttot, label="{0}: Total".format(key),
                 color=color[i], linestyle='solid', marker=next(marker))
        plt.plot(range(1, 11), [tpca] * 10, label="{0}: PCA".format(key),
                 color=color[i], linestyle='dashed', alpha=0.5)
        plt.plot(range(1, 11), ttrain, label="{0}: Training".format(key),
                 color=color[i], linestyle='dotted', alpha=0.5)
        plt.plot(range(1, 11), ttest, label="{0}: Testing".format(key),
                 color=color[i], linestyle='dashdot', alpha=0.5)

    lgd = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.title(title)
    plt.xlabel("$k$")
    plt.ylabel("Run Time (s)")
    save_path = os.path.join(os.path.dirname(__file__), out)
    plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def main():
    # first we just need a loader for all the data, which is in the
    # folder given below
    mndata = MNIST('data/mnist')

    # load the training data, then wrap in numpy arrays to make it
    # easier to work with
    images, labels = mndata.load_training()
    X_train = np.array(images)
    y_train = np.array(labels)

    # Problem 3a
    X_0_train = X_train[y_train == 0]
    pca_plot(X_0_train, desc="MNIST 0", out='hw2/pca_0.pdf')

    # Problem 3b
    X_1_train = X_train[y_train == 1]
    pca_plot(X_1_train, desc="MNIST 1", out='hw2/pca_1.pdf')

    # Problem 3c
    pca_plot((X_0_train, X_1_train), desc="MNIST 0, 1", out='hw2/pca_01.pdf')

    # Problem 4
    # load usps data
    print("--- USPS Data ---")
    print("--- kNN ---")
    data = pd.read_csv('data/usps/zip.train',
                       header=None, delimiter=' ').iloc[:, :-1]
    y_train = data.pop(0).values
    X_train = data.values

    data = pd.read_csv('data/usps/zip.test', header=None, delimiter=' ')
    y_test = data.pop(0).values
    X_test = data.values

    X = (X_train, X_test)
    y = (y_train, y_test)

    keys = ["50% Variance", "95% Variance", "No PCA"]
    errors_50, times_50 = pca_knn(X, y, var=.5)
    errors_95, times_95 = pca_knn(X, y, var=.95)
    errors_none, times_none = pca_knn(X, y)

    errors = {keys[0]: errors_50,
              keys[1]: errors_95,
              keys[2]: errors_none
              }

    times = {keys[0]: times_50,
             keys[1]: times_95,
             keys[2]: times_none
             }

    min_val = np.min([errors[k] for k in errors.keys()])
    result = [k for k in errors.keys() if min_val in errors[k]][0]
    idx = np.where(errors[result] == min_val)[0][0] + 1
    print("min error = {0}, label = {1}, k = {2}".format(min_val, result, idx))

    plot_results(errors, title="$k$NN: Error Rate vs. $k$ Neighbors",
                 out='hw2/knn_errors.pdf')

    min_time = np.inf
    min_key = ""
    min_k = 0
    for k in times.keys():
        tpca, ttrain, ttest = times[k]
        ttots = tpca + ttrain + ttest
        if np.min(ttots) < min_time:
            min_time = np.min(ttots)
            min_key = k
            min_k = np.where(ttots == min_time)[0][0]

    print("min time = {0}, label = {1}, k = {2}".format(min_time,
                                                        min_key, min_k + 1))

    plot_times(times, title="$k$NN: Run Time vs. $k$ Neighbors",
               out='hw2/knn_times.pdf')

    # Problem 5
    print("--- NLC ---")
    errors_50, times_50 = pca_nlc(X, y, var=.5)
    errors_95, times_95 = pca_nlc(X, y, var=.95)
    errors_none, times_none = pca_nlc(X, y)

    errors = {keys[0]: errors_50,
              keys[1]: errors_95,
              keys[2]: errors_none
              }

    times = {keys[0]: times_50,
             keys[1]: times_95,
             keys[2]: times_none
             }

    min_val = np.min([errors[k] for k in errors.keys()])
    result = [k for k in errors.keys() if min_val in errors[k]][0]
    idx = np.where(errors[result] == min_val)[0][0] + 1
    print("min error = {0}, label = {1}, k = {2}".format(min_val, result, idx))

    plot_results(errors,
                 title="NLC: Error Rate vs. $k$ Neighbors per Centroid",
                 out='hw2/nlc_errors.pdf')

    min_time = np.inf
    min_key = ""
    min_k = 0
    for k in times.keys():
        tpca, ttrain, ttest = times[k]
        ttots = tpca + ttrain + ttest
        if np.min(ttots) < min_time:
            min_time = np.min(ttots)
            min_key = k
            min_k = np.where(ttots == min_time)[0][0]

    print("min time = {0}, label = {1}, k = {2}".format(min_time,
                                                        min_key, min_k + 1))

    plot_times(times, title="NLC: Run Time vs. $k$ Neighbors per Centroid",
               out='hw2/nlc_times.pdf')

    # Problem 6
    print("--- SDSS Data ---")
    photo_data = pd.read_csv('data/photo.data')
    y = photo_data.pop('class').values
    X = np.array([(photo_data['u'] - photo_data['g']).values,
                  (photo_data['g'] - photo_data['r']).values,
                  (photo_data['r'] - photo_data['i']).values,
                  (photo_data['i'] - photo_data['z']).values]).T

    # split into training/testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    X = (X_train, X_test)
    y = (y_train, y_test)

    print("--- SDSS kNN ---")

    keys = ["50% Variance", "95% Variance", "No PCA"]
    errors_50, times_50 = pca_knn(X, y, var=.5)
    errors_95, times_95 = pca_knn(X, y, var=.95)
    errors_none, times_none = pca_knn(X, y)

    errors = {keys[0]: errors_50,
              keys[1]: errors_95,
              keys[2]: errors_none
              }

    times = {keys[0]: times_50,
             keys[1]: times_95,
             keys[2]: times_none
             }

    min_val = np.min([errors[k] for k in errors.keys()])
    result = [k for k in errors.keys() if min_val in errors[k]][0]
    idx = np.where(errors[result] == min_val)[0][0] + 1
    print("min error = {0}, label = {1}, k = {2}".format(min_val, result, idx))

    plot_results(errors, title="$k$NN: Error Rate vs. $k$ Neighbors",
                 out='hw2/sdss_knn_errors.pdf')

    min_time = np.inf
    min_key = ""
    min_k = 0
    for k in times.keys():
        tpca, ttrain, ttest = times[k]
        ttots = tpca + ttrain + ttest
        if np.min(ttots) < min_time:
            min_time = np.min(ttots)
            min_key = k
            min_k = np.where(ttots == min_time)[0][0]

    print("min time = {0}, label = {1}, k = {2}".format(min_time,
                                                        min_key, min_k + 1))

    plot_times(times, title="$k$NN: Run Time vs. $k$ Neighbors",
               out='hw2/sdss_knn_times.pdf')

    # Problem 5
    print("--- SDSS NLC ---")
    errors_50, times_50 = pca_nlc(X, y, var=.5)
    errors_95, times_95 = pca_nlc(X, y, var=.95)
    errors_none, times_none = pca_nlc(X, y)

    errors = {keys[0]: errors_50,
              keys[1]: errors_95,
              keys[2]: errors_none
              }

    times = {keys[0]: times_50,
             keys[1]: times_95,
             keys[2]: times_none
             }

    min_val = np.min([errors[k] for k in errors.keys()])
    result = [k for k in errors.keys() if min_val in errors[k]][0]
    idx = np.where(errors[result] == min_val)[0][0] + 1
    print("min error = {0}, label = {1}, k = {2}".format(min_val, result, idx))

    plot_results(errors,
                 title="NLC: Error Rate vs. $k$ Neighbors per Centroid",
                 out='hw2/sdss_nlc_errors.pdf')

    min_time = np.inf
    min_key = ""
    min_k = 0
    for k in times.keys():
        tpca, ttrain, ttest = times[k]
        ttots = tpca + ttrain + ttest
        if np.min(ttots) < min_time:
            min_time = np.min(ttots)
            min_key = k
            min_k = np.where(ttots == min_time)[0][0]

    print("min time = {0}, label = {1}, k = {2}".format(min_time,
                                                        min_key, min_k + 1))

    plot_times(times, title="NLC: Run Time vs. $k$ Neighbors per Centroid",
               out='hw2/sdss_nlc_times.pdf')


if __name__ == '__main__':
    font = {'size': 20}
    mpl.rc('font', **font)
    main()
