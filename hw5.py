#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# <Brief overview of the package>
# Copyright (C) 2017-2018  J. Andrew Casey-Clyde
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Sat Nov  3 13:16:50 2018

@author: jacaseyclyde
"""
import os

import numpy as np

import pandas as pd
from scipy.io import loadmat
from mnist import MNIST

import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.multiclass import OneVsOneClassifier as OVO


def convergence(x):
    guesses = np.array([x])
    while True:
        guesses = np.append(guesses,
                            guesses[-1] - ((np.cos(guesses[-1]) - guesses[-1])
                                           / (-1 * np.sin(guesses[-1]) - 1)))

        if (guesses[-1] == guesses[-2]):
            return guesses


def grad_descent(x, rate=0.05):
    guesses = np.array([x])
    i = 0
    while True:
        guesses = np.append(guesses,
                            guesses[-1] - rate * (guesses[-1] ** 3
                                                  - 9 * guesses[-1] ** 2 + 7))

        i += 1
        if (guesses[-1] == guesses[-2] or i == 30):
            return guesses


def convergence_plot(series, title, out):
    plt.figure(figsize=(12, 12))
    marker = itertools.cycle(('o', '^', '+'))
    for key in series:
        plt.plot(range(len(series[key])), series[key],
                 label=key, marker=next(marker))

    plt.legend()
    plt.title(title)
    save_path = os.path.join(os.path.dirname(__file__), out)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def boundary_plot(X, y, clf, title, out):
    X1 = X[y == 1]
    X2 = X[y == 2]

    plt.figure(figsize=(12, 12))
    plt.plot(X1[:, 0], X1[:, 1], 'o', color='red', label='1')
    plt.plot(X2[:, 0], X2[:, 1], 'o', color='blue', label='2')

    nx, ny = 200, 200
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    save_path = os.path.join(os.path.dirname(__file__), out)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def main():
    # PROBLEM 1
    convergences = {"0": convergence(0),
                    "1": convergence(1),
                    "$\\pi/2$": convergence(np.pi / 2)}
    convergence_plot(convergences, "Newton's method approximation",
                     'hw5/newton.pdf')

    # PROBLEM 2
    rates = {"0.01": grad_descent(0, 0.01),
             "0.05": grad_descent(0, 0.05),
             "0.1": grad_descent(0, 0.1)}
    convergence_plot(rates, "Gradient Descent about $f(0)$",
                     'hw5/gradient_descent.pdf')

    # PROBLEM 3
    gaussians = loadmat('data/twogaussians/twogaussians.mat')
    X_train = gaussians['Xtr']
    y_train = np.ravel(gaussians['ytr'])

    X_test = gaussians['Xtst']
    y_test = np.ravel(gaussians['ytst'])

    lr = LR(solver='lbfgs', max_iter=1000).fit(X_train, y_train)
    lr_err_rate = 1 - lr.score(X_test, y_test)
    title = "LR Decision Boundary (Error Rate = {0:.4f})".format(lr_err_rate)
    boundary_plot(X_train, y_train, lr,
                  title, 'hw5/lr_decision_boundary.pdf')

    lda = LDA().fit(X_train, y_train)
    lda_err_rate = 1 - lda.score(X_test, y_test)
    title = "LDA Decision Boundary (Error Rate = {0:.4f})".format(lda_err_rate)
    boundary_plot(X_train, y_train, lda,
                  title, 'hw5/lda_decision_boundary.pdf')

    # PROBLEM 4
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

    # extra credit problem 6. uses USPS data before PCA
    mlt = LR(multi_class='multinomial',
             solver='lbfgs', max_iter=5000).fit(X_train, y_train)
    l1_multi_err_rate = 1 - mlt.score(X_test, y_test)

    pca = PCA(n_components=.95)
    pca.fit(X_train)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    ovo = OVO(LR(solver='lbfgs', max_iter=5000)).fit(X_train, y_train)
    lr_ovo_err_rate = 1 - ovo.score(X_test, y_test)

    ovr = LR(multi_class='ovr',
             solver='lbfgs', max_iter=5000).fit(X_train, y_train)
    lr_ovr_err_rate = 1 - ovr.score(X_test, y_test)

    mlt = LR(multi_class='multinomial',
             solver='lbfgs', max_iter=5000).fit(X_train, y_train)
    lr_multi_err_rate = 1 - mlt.score(X_test, y_test)

    lda = LDA().fit(X_train, y_train)
    lda_err_rate = 1 - lda.score(X_test, y_test)

    print("=== USPS Data ===")
    print("LR OVO Error Rate = {0:.4f}".format(lr_ovo_err_rate))
    print("LR OVR Error Rate = {0:.4f}".format(lr_ovr_err_rate))
    print("LR Multinomial Error Rate = {0:.4f}".format(lr_multi_err_rate))
    print("LDA Error Rate = {0:.4f}".format(lda_err_rate))
    print("LR L1 Regularized Multinomial "
          "Error Rate = {0:.4f}\n".format(l1_multi_err_rate))

    # PROBLEM 5
    # read in mnist data
    mndata = MNIST('data/mnist')
    print("=== MNIST Data ===")

    # training data
    images, labels = mndata.load_training()
    X_train = np.array(images)
    y_train = np.array(labels)

    # test data
    images, labels = mndata.load_testing()
    X_test = np.array(images)
    y_test = np.array(labels)

    subsets = {"0, 1": [0, 1],
               "4, 9": [4, 9],
               "0, 1, 2": [0, 1, 2],
               "3, 5, 8": [3, 5, 8]}

    for digits in subsets:
        X_train_sub = X_train[np.isin(y_train, subsets[digits])]
        y_train_sub = y_train[np.isin(y_train, subsets[digits])]

        X_test_sub = X_test[np.isin(y_test, subsets[digits])]
        y_test_sub = y_test[np.isin(y_test, subsets[digits])]

        pca = PCA(n_components=.95)
        pca.fit(X_train_sub)

        X_train_sub = pca.transform(X_train_sub)
        X_test_sub = pca.transform(X_test_sub)

        ovo = OVO(LR(solver='lbfgs', max_iter=5000)).fit(X_train_sub,
                                                         y_train_sub)
        lr_ovo_err_rate = 1 - ovo.score(X_test_sub, y_test_sub)

        ovr = LR(multi_class='ovr',
                 solver='lbfgs', max_iter=5000).fit(X_train_sub,
                                                    y_train_sub)
        lr_ovr_err_rate = 1 - ovr.score(X_test_sub, y_test_sub)

        mlt = LR(multi_class='multinomial',
                 solver='lbfgs', max_iter=5000).fit(X_train_sub,
                                                    y_train_sub)
        lr_multi_err_rate = 1 - mlt.score(X_test_sub, y_test_sub)

        lda = LDA().fit(X_train_sub, y_train_sub)
        lda_err_rate = 1 - lda.score(X_test_sub, y_test_sub)

        print("=== {0} ===".format(digits))
        print("LR OVO Error Rate = {0:.4f}".format(lr_ovo_err_rate))
        print("LR OVR Error Rate = {0:.4f}".format(lr_ovr_err_rate))
        print("LR Multinomial Error Rate = {0:.4f}".format(lr_multi_err_rate))
        print("LDA Error Rate = {0:.4f}".format(lda_err_rate))
        print("\n")


if __name__ == '__main__':
    font = {'size': 24}
    mpl.rc('font', **font)
    directory = os.path.join(os.path.dirname(__file__), 'hw5')
    if not os.path.exists(directory):
        os.makedirs(directory)
#    plt.xkcd()
    main()
