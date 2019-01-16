#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 09:20:18 2018

@author: jacaseyclyde
"""

import os

import numpy as np
import pandas as pd

from scipy.io import loadmat

import matplotlib as mpl
import matplotlib.pyplot as plt

from mnist import MNIST

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def main():
    # PROBLEMS 1, 2, 3 (USPS Data)
    ####################################################################
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

    errs = None
    rows = np.array([])
    cols = np.array([])

    # apply pca
    for var in [.95, .96, .97, .98, .99, 1.]:
        if var == 1:
            pca = PCA()
        else:
            pca = PCA(n_components=var)

        pca.fit(X_train)

        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        n_comps = pca.n_components_
        rows = np.append(rows,
                         "{0:.0f} components ({1:.0f}% variance)"
                         .format(n_comps, var * 100))

        # setting this up in here so that the classifiers re-initialize
        # for each variance
        classifiers = {"$k$NN ($k$ = 3)": KNeighborsClassifier(n_neighbors=3),
                       "LDA": LDA()}

        var_errs = np.array([])
        for key in classifiers:
            # check first if we already have all our classifier columns
            if (len(cols) < len(classifiers)):
                cols = np.append(cols, key)

            cls = classifiers[key]
            cls_err = 1 - cls.fit(X_train_pca,
                                  y_train).score(X_test_pca, y_test)
            var_errs = np.append(var_errs, "{0:.4f}".format(cls_err))
        if errs is not None:
            errs = np.vstack((errs, var_errs))
        else:
            errs = np.array([var_errs])

    # make a table of the results
    plt.figure(figsize=(12, 2))
    plt.table(cellText=errs, rowLabels=rows, colLabels=cols,
              loc='upper left')
    plt.axis('off')
    save_path = os.path.join(os.path.dirname(__file__),
                             'hw4/usps_err_rates_lda.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.title("USPS Error Rates, LDA")
    plt.show()

    errs = None
    rows = np.array([])
    cols = np.array([])

    # apply pca
    for var in [.95, .96, .97, .98, .99, 1.]:
        if var == 1:
            pca = PCA()
        else:
            pca = PCA(n_components=var)

        pca.fit(X_train)

        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        n_comps = pca.n_components_
        rows = np.append(rows,
                         "{0:.0f} components ({1:.0f}% variance)"
                         .format(n_comps, var * 100))

        # setting this up in here so that the classifiers re-initialize
        # for each variance
        classifiers = {"$k$NN ($k$ = 3)": KNeighborsClassifier(n_neighbors=3),
                       "QDA": QDA()}

        var_errs = np.array([])
        for key in classifiers:
            # check first if we already have all our classifier columns
            if (len(cols) < len(classifiers)):
                cols = np.append(cols, key)

            cls = classifiers[key]
            cls_err = 1 - cls.fit(X_train_pca,
                                  y_train).score(X_test_pca, y_test)
            var_errs = np.append(var_errs, "{0:.4f}".format(cls_err))
        if errs is not None:
            errs = np.vstack((errs, var_errs))
        else:
            errs = np.array([var_errs])

    plt.figure(figsize=(12, 2))
    plt.table(cellText=errs, rowLabels=rows, colLabels=cols,
              loc='upper left')
    plt.axis('off')
    save_path = os.path.join(os.path.dirname(__file__),
                             'hw4/usps_err_rates_qda.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.title("USPS Error Rates, QDA")
    plt.show()

    errs = None
    rows = np.array([])
    cols = np.array([])

    # apply pca
    for var in [.95, .96, .97, .98, .99, 1.]:
        if var == 1:
            pca = PCA()
        else:
            pca = PCA(n_components=var)

        pca.fit(X_train)

        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        n_comps = pca.n_components_
        rows = np.append(rows,
                         "{0:.0f} components ({1:.0f}% variance)"
                         .format(n_comps, var * 100))

        # setting this up in here so that the classifiers re-initialize
        # for each variance
        classifiers = {"$k$NN ($k$ = 3)": KNeighborsClassifier(n_neighbors=3),
                       "GNB": GaussianNB()}

        var_errs = np.array([])
        for key in classifiers:
            # check first if we already have all our classifier columns
            if (len(cols) < len(classifiers)):
                cols = np.append(cols, key)

            cls = classifiers[key]
            cls_err = 1 - cls.fit(X_train_pca,
                                  y_train).score(X_test_pca, y_test)
            var_errs = np.append(var_errs, "{0:.4f}".format(cls_err))
        if errs is not None:
            errs = np.vstack((errs, var_errs))
        else:
            errs = np.array([var_errs])

    plt.figure(figsize=(12, 2))
    plt.table(cellText=errs, rowLabels=rows, colLabels=cols,
              loc='upper left')
    plt.axis('off')
    save_path = os.path.join(os.path.dirname(__file__),
                             'hw4/usps_err_rates_gnb.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.title("USPS Error Rates, GNB")
    plt.show()

    # PROBLEM 4 (MNIST Data)
    ####################################################################
    # read in mnist data
    mndata = MNIST('data/mnist')

    # training data
    images, labels = mndata.load_training()
    X_train = np.array(images)
    y_train = np.array(labels)

    # test data
    images, labels = mndata.load_testing()
    X_test = np.array(images)
    y_test = np.array(labels)

    errs = None
    rows = np.array([])
    cols = np.array([])

    classifiers = {"LDA": LDA(),
                   "QDA": QDA()}
    variances = [.95, .96, .97, .98, .99, 1.]
    subsets = {"0, 1": [0, 1],
               "4, 9": [4, 9],
               "0, 1, 2": [0, 1, 2],
               "3, 5, 8": [3, 5, 8]}
    for components in variances:
        for cls_key in classifiers:
            var_errs = np.array([])
            for i, key in enumerate(subsets):
                if components == 1:
                    pca = PCA()
                else:
                    pca = PCA(n_components=components)

                X_train_sub = X_train[np.isin(y_train, subsets[key])]
                y_train_sub = y_train[np.isin(y_train, subsets[key])]

                X_test_sub = X_test[np.isin(y_test, subsets[key])]
                y_test_sub = y_test[np.isin(y_test, subsets[key])]

                pca.fit(X_train_sub)

                X_train_pca = pca.transform(X_train_sub)
                X_test_pca = pca.transform(X_test_sub)

                n_comps = pca.n_components_

                # only need to fill these if they aren't already filled
                if (i == 0):
                    rows = np.append(rows,
                                     "{0:.0f} components ({1:.0f}% variance) "
                                     "+ {2}"
                                     .format(n_comps,
                                             components * 100, cls_key))

                if (len(cols) < len(subsets)):
                    cols = np.append(cols, key)

                cls = classifiers[cls_key]
                cls.fit(X_train_pca, y_train_sub)
                cls_err = 1 - cls.score(X_test_pca, y_test_sub)
                var_errs = np.append(var_errs, "{0:.4f}".format(cls_err))

            if errs is not None:
                errs = np.vstack((errs, var_errs))
            else:
                errs = np.array([var_errs])

    # make a table of the results
    plt.figure(figsize=(12, 2))
    plt.table(cellText=errs, rowLabels=rows, colLabels=cols,
              loc='upper left')
    plt.axis('off')
    save_path = os.path.join(os.path.dirname(__file__),
                             'hw4/mnist_err_rates.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.title("MNIST Error Rates")
    plt.show()

    # PROBLEM 5
    ###################################################################
    gaussians = loadmat('data/twogaussians/twogaussians.mat')
    X_train = gaussians['Xtr']
    y_train = np.ravel(gaussians['ytr'])

    X_test = gaussians['Xtst']
    y_test = np.ravel(gaussians['ytst'])

    lda = LDA()
    lda.fit(X_train, y_train)
    lda_err_rate = 1 - lda.score(X_test, y_test)
    print('{0:.4f}'.format(lda_err_rate))

    qda = QDA()
    qda.fit(X_train, y_train)
    qda_err_rate = 1 - qda.score(X_test, y_test)
    print('{0:.4f}'.format(qda_err_rate))

    X1 = X_train[y_train == 1]
    X2 = X_train[y_train == 2]

    plt.figure(figsize=(12, 12))
    plt.plot(X1[:, 0], X1[:, 1], 'o', color='red', label='1')
    plt.plot(X2[:, 0], X2[:, 1], 'o', color='blue', label='2')

    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    save_path = os.path.join(os.path.dirname(__file__),
                             'hw4/lda_decision_boundary.pdf')
    plt.title("LDA Decision Boundary (Error Rate = {0:.4f})"
              .format(lda_err_rate))
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.plot(X1[:, 0], X1[:, 1], 'o', color='red', label='1')
    plt.plot(X2[:, 0], X2[:, 1], 'o', color='blue', label='2')

    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = qda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    save_path = os.path.join(os.path.dirname(__file__),
                             'hw4/qda_decision_boundary.pdf')
    plt.title("QDA Decision Boundary (Error Rate = {0:.4f})"
              .format(qda_err_rate))
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    font = {'size': 24}
    mpl.rc('font', **font)
    directory = os.path.join(os.path.dirname(__file__), 'hw4')
    if not os.path.exists(directory):
        os.makedirs(directory)
#    plt.xkcd()
    main()
