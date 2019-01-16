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
Created on Sat Nov  21 14:52:37 2018

@author: jacaseyclyde
"""
import os

import numpy as np

from scipy.io import loadmat
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from sklearn.linear_model import LogisticRegression as LR

from sklearn.svm import LinearSVC
from sklearn.svm import SVC


def main():
    # load data
    gaussians = loadmat('data/twogaussians/twogaussians.mat')
    X_train = gaussians['Xtr']
    y_train = np.ravel(gaussians['ytr'])

    X_test = gaussians['Xtst']
    y_test = np.ravel(gaussians['ytst'])

    print("=== Problem 1 ===")
    print("=== a ===")
    C_vals = 2. ** np.arange(-6, 5)

    err_means = np.array([])
    err_stds = np.array([])
    for C in C_vals:
        svm = LinearSVC(C=C, multi_class='ovr')
        scores = cross_val_score(svm, X_train, y_train,
                                 cv=5, n_jobs=-1)

        err_means = np.append(err_means, 1 - np.mean(scores))
        err_stds = np.append(err_stds, np.std(scores))

    plt.figure(figsize=(12, 12))
    plt.errorbar(np.log2(C_vals), err_means, yerr=err_stds, capsize=2)
    plt.ylim(bottom=0)
    plt.xlabel("$\\log_{2}(C)$")
    plt.ylabel("Average Error Rate")
    plt.title("Linear SVM: Average Error rate vs $C$, 5-fold cross validation")
    save_path = os.path.join(os.path.dirname(__file__), 'hw6/linear_svm.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    low_idxs = err_means == np.min(err_means)
    if len(low_idxs) > 1:
        C_best = C_vals[err_means == np.min(err_means)]
        low_idx = (err_stds == np.min(err_stds[low_idxs]))
    else:
        low_idx = low_idxs

    C_best = C_vals[low_idx]
    print("C_best = 2 ** {0:.4f}".format(np.log2(C_best[0])))
    print("err = {0:.4f} (+/-{1:.4f})".format(err_means[low_idx][0],
                                              err_stds[low_idx][0]))

    print("=== b ===")
    X1 = X_test[y_test == 1]
    X2 = X_test[y_test == 2]

    plt.figure(figsize=(12, 12))
    plt.plot(X1[:, 0], X1[:, 1], 'o', color='red', label='1')
    plt.plot(X2[:, 0], X2[:, 1], 'o', color='blue', label='2')

    classifiers = {"SVM": LinearSVC(C=C_best, multi_class='ovr'),
                   "LDA": LDA(),
                   "Linear Regression": LR(solver='lbfgs', max_iter=1000)}
    colors = itertools.cycle(('k', 'g', 'orange'))

    for key in classifiers:
        print("== {0} ==".format(key))
        clf = classifiers[key]
        clf.fit(X_train, y_train)

        nx, ny = 200, 200
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                             np.linspace(y_min, y_max, ny))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, [.5], colors=next(colors), linewidths=2.)

        err_rate = 1 - clf.score(X_test, y_test)
        print(err_rate)

    plt.title("Linear Decision Boundaries")

    svm_pos = plt.Line2D((0, 1), (0, 0), color='k', linestyle='-', linewidth=2)
    lda_pos = plt.Line2D((0, 1), (0, 0), color='g', linestyle='-', linewidth=2)
    lr_pos = plt.Line2D((0, 1), (0, 0), color='orange', linestyle='-',
                        linewidth=2)

    plt.legend([svm_pos, lda_pos, lr_pos],
               ['SVM', 'LDA', 'Linear Regression'])

    save_path = os.path.join(os.path.dirname(__file__),
                             'hw6/linear_decision_boundary.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    print("=== Problem 2 ===")
    print("=== a ===")
    C_vals = 2. ** np.arange(-6, 5)

    err_means = np.array([])
    err_stds = np.array([])

    for C in C_vals:
        svm = SVC(C=C, decision_function_shape='ovr', kernel='poly',
                  gamma='scale', degree=2)
        scores = cross_val_score(svm, X_train, y_train,
                                 cv=5, n_jobs=-1)

        err_means = np.append(err_means, 1 - np.mean(scores))
        err_stds = np.append(err_stds, np.std(scores))

    plt.figure(figsize=(12, 12))
    plt.errorbar(np.log2(C_vals), err_means, yerr=err_stds)
    plt.ylim(bottom=0)
    plt.xlabel("\\log_{2}(C)$")
    plt.ylabel("Average Error Rate")
    plt.title("Linear SVM: Average Error rate vs $C$, 5-fold cross validation")
    save_path = os.path.join(os.path.dirname(__file__), 'hw6/quad_svm.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    low_idxs = err_means == np.min(err_means)
    if len(low_idxs) > 1:
        C_best = C_vals[err_means == np.min(err_means)]
        low_idx = (err_stds == np.min(err_stds[low_idxs]))
    else:
        low_idx = low_idxs

    C_best = C_vals[low_idx]
    print("C_best = 2 ** {0:.4f}".format(np.log2(C_best[0])))
    print("err = {0:.4f} (+/-{1:.4f})".format(err_means[low_idx][0],
                                              err_stds[low_idx][0]))

    print("=== b ===")
    plt.figure(figsize=(12, 12))
    plt.plot(X1[:, 0], X1[:, 1], 'o', color='red', label='1')
    plt.plot(X2[:, 0], X2[:, 1], 'o', color='blue', label='2')

    classifiers = {"SVM": SVC(C=C_best, decision_function_shape='ovr',
                              kernel='poly', gamma='scale', degree=2),
                   "QDA": QDA()}
    colors = itertools.cycle(('k', 'g', 'orange'))

    for key in classifiers:
        print("== {0} ==".format(key))
        clf = classifiers[key]
        clf.fit(X_train, y_train)

        nx, ny = 200, 200
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                             np.linspace(y_min, y_max, ny))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, [.5], colors=next(colors), linewidths=2.)

        err_rate = 1 - clf.score(X_test, y_test)
        print(err_rate)

    plt.title("Quadratic Decision Boundaries")

    svm_pos = plt.Line2D((0, 1), (0, 0), color='k', linestyle='-', linewidth=2)
    lda_pos = plt.Line2D((0, 1), (0, 0), color='g', linestyle='-', linewidth=2)

    plt.legend([svm_pos, lda_pos],
               ['SVM', 'QDA'])

    save_path = os.path.join(os.path.dirname(__file__),
                             'hw6/quad_decision_boundary.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

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

    subsets = {"0, 1": [0, 1],
               "1, 7": [1, 7],
               "4, 9": [4, 9]}

    kernels = ['linear', 'poly', 'poly']
    kwargs = [{}, {'degree': 2}, {'degree': 3}]

    print("=== Problem 3 ===")
    for digits in subsets:
        print("=== {0} ===".format(digits))
        subset = subsets[digits]
        X_train_sub = X_train[np.isin(y_train, subset)]
        y_train_sub = y_train[np.isin(y_train, subset)]

        X_test_sub = X_test[np.isin(y_test, subset)]
        y_test_sub = y_test[np.isin(y_test, subset)]

        pca = PCA(n_components=.95)
        pca.fit(X_train_sub)

        X_train_sub = pca.transform(X_train_sub)
        X_test_sub = pca.transform(X_test_sub)

        plt.figure(figsize=(12, 12))
        for kernel, kw in zip(kernels, kwargs):
            errs = np.array([])
            for C in C_vals:
                svm = SVC(C=C, decision_function_shape='ovr', kernel=kernel,
                          gamma='scale', **kw)
                svm.fit(X_train_sub, y_train_sub)

                errs = np.append(errs, 1 - svm.score(X_test_sub, y_test_sub))

            if kernel == 'linear':
                label = "Linear"
            else:
                if kw['degree'] == 2:
                    label = "Quad. Poly."
                else:
                    label = "Cubic Poly."

            plt.plot(np.log2(C_vals), errs, label=label)

        plt.ylim(bottom=0)
        plt.xlabel("$\\log_{2}(C)$")
        plt.ylabel("Error Rate")
        plt.title("USPS: Error Rates vs. $C$, [{0}]".format(digits))
        plt.legend()
        save_path = os.path.join(os.path.dirname(__file__),
                                 'hw6/usps_{0}{1}.pdf'.format(subset[0],
                                                              subset[1]))
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    print("=== Problem 4 ===")
    # doing both over the same range for now
    params = {'C': C_vals,
              'gamma': C_vals}
    svm = SVC(decision_function_shape='ovr', kernel='rbf')
    for digits in subsets:
        print("=== {0} ===".format(digits))
        X_train_sub = X_train[np.isin(y_train, subsets[digits])]
        y_train_sub = y_train[np.isin(y_train, subsets[digits])]

        X_test_sub = X_test[np.isin(y_test, subsets[digits])]
        y_test_sub = y_test[np.isin(y_test, subsets[digits])]

        pca = PCA(n_components=.95)
        pca.fit(X_train_sub)

        X_train_sub = pca.transform(X_train_sub)
        X_test_sub = pca.transform(X_test_sub)

        clf = GridSearchCV(svm, params, cv=5, n_jobs=-1)
        clf.fit(X_train_sub, y_train_sub)

        print("Best parameters: {0}".format(clf.best_params_))

        print("C_best = {0}".format(np.log2(clf.best_params_['C'])))
        print("sigma_best = {0}".format(np.log2(clf.best_params_['gamma'])))
        print("Best Error Rate: {0:.4f}".format(1 - clf.best_score_))

        y_true, y_pred = y_test_sub, clf.predict(X_test_sub)
        conf_mat = confusion_matrix(y_true, y_pred)

        print("Test Error Rate: {0:.4f}".format(1
                                                - np.sum(np.diagonal(conf_mat))
                                                / np.sum(conf_mat)))


if __name__ == '__main__':
    font = {'size': 24}
    mpl.rc('font', **font)
    figsize = (12, 12)
    directory = os.path.join(os.path.dirname(__file__), 'hw6')
    if not os.path.exists(directory):
        os.makedirs(directory)
#    plt.xkcd()
    main()
