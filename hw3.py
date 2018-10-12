#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:21:20 2018

@author: jacaseyclyde
"""
import os
import sys

from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
    plt.show()

    save_path = os.path.join(os.path.dirname(__file__), out)
    plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')


def pca_lda_knn(X, y, lda_flag=True):
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
    for k in tqdm(np.arange(1, kmax + 1), desc=desc, file=sys.stdout):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        errors = np.append(errors, 1 - knn.score(X_test, y_test))

    return errors


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
          min_key, min_k + 1))


def main():
    # Problem 1
    # load iris data
    iris_data_path = os.path.join(os.path.dirname(__file__),
                                  'data', 'iris.data')
    iris_data = pd.read_csv(iris_data_path)
    y = iris_data.pop("species").values
    X = iris_data.values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X_pca, y)

    pca_lda_plot(X_lda, y, components=2,
                 title="2 Component PCA + LDA", out='hw3/pca_lda_2c.pdf')

    # Problem 3
    data = pd.read_csv('data/usps/zip.train',
                       header=None, delimiter=' ').iloc[:, :-1]
    y_train = data.pop(0).values
    X_train = data.values

    # 3a
    X_0 = X_train[y_train == 0]
    y_0 = y_train[y_train == 0]

    X_1 = X_train[y_train == 1]
    y_1 = y_train[y_train == 1]

    X_01 = np.append(X_0, X_1, axis=0)
    y_01 = np.append(y_0, y_1, axis=0)

    pca_lda_plot(X_01, y_01,
                 title="95% PCA + LDA, USPS 0, 1",
                 out='hw3/pca_lda_01.pdf')

    # 3b
    X_4 = X_train[y_train == 4]
    y_4 = y_train[y_train == 4]

    X_9 = X_train[y_train == 9]
    y_9 = y_train[y_train == 9]

    X_49 = np.append(X_4, X_9, axis=0)
    y_49 = np.append(y_4, y_9, axis=0)

    pca_lda_plot(X_49, y_49,
                 title="95% PCA + LDA, USPS 4, 9",
                 out='hw3/pca_lda_49.pdf')

    # 3c
    X_2 = X_train[y_train == 2]
    y_2 = y_train[y_train == 2]

    X_3 = X_train[y_train == 3]
    y_3 = y_train[y_train == 3]

    X_123 = np.append(np.append(X_1, X_2, axis=0), X_3, axis=0)
    y_123 = np.append(np.append(y_1, y_2, axis=0), y_3, axis=0)

    pca_lda_plot(X_123, y_123,
                 title="95% PCA + LDA, USPS 1, 2, 3",
                 out='hw3/pca_lda_123.pdf')

    X_5 = X_train[y_train == 5]
    y_5 = y_train[y_train == 5]

    X_8 = X_train[y_train == 8]
    y_8 = y_train[y_train == 8]

    X_358 = np.append(np.append(X_3, X_5, axis=0), X_8, axis=0)
    y_358 = np.append(np.append(y_3, y_5, axis=0), y_8, axis=0)

    pca_lda_plot(X_358, y_358,
                 title="95% PCA + LDA, USPS 3, 5, 8",
                 out='hw3/pca_lda_358.pdf')

    # Problem 4
    data = pd.read_csv('data/usps/zip.test', header=None, delimiter=' ')
    y_test = data.pop(0).values
    X_test = data.values

    keys = ["PCA + LDA", "PCA"]
    errors = {keys[0]: pca_lda_knn((X_train, X_test), (y_train, y_test),
                                   lda_flag=True),
              keys[1]: pca_lda_knn((X_train, X_test), (y_train, y_test),
                                   lda_flag=False)}

    plot_results(errors, title="$k$NN: Error rate vs. $k$ Neighbors",
                 out='hw3/knn_errors.pdf')

    min_vals(errors, label="Errors")


if __name__ == '__main__':
    font = {'size': 24}
    mpl.rc('font', **font)
    directory = os.path.join(os.path.dirname(__file__), 'hw3')
    if not os.path.exists(directory):
        os.makedirs(directory)
#    plt.xkcd()
    main()
