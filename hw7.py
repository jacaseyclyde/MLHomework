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
# pylint: disable=C0103
import os
import sys
import itertools

import numpy as np

from scipy.stats import norm

import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression as LR
from sklearn.multiclass import OneVsOneClassifier as OVO
from sklearn.multiclass import OneVsRestClassifier as OVR

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.svm import SVC

from tqdm import tqdm


C_VALS = 2. ** np.arange(-4, 6)
SIGMA = 1.


def err_plot(errs, labels, lr=None, knn=None, title="Title", out='out.pdf'):
    plt.figure()
    for err, label in zip(errs, labels):
        plt.plot(np.log2(C_VALS), err, label=label)
    if lr is not None:
        plt.axhline(lr, label="LR",
                    alpha=0.7, linestyle='--', color='r')
    if knn is not None:
        colors = itertools.cycle(('k', 'gray', 'silver'))
        linestyles = itertools.cycle(('solid', 'dashed',
                                      'dashdot', 'dotted'))
        for k, err in knn:
            label = "kNN, k = {0}".format(k)
            plt.axhline(err, color=next(colors),
                        linestyle=next(linestyles), label=label)
    plt.xlabel("$\\log_{2}(C)$")
    plt.ylabel("Error Rate")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    save_path = os.path.join(os.path.dirname(__file__), out)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def gaussian(dist):
    return norm.pdf(dist, loc=dist[0], scale=SIGMA)


def main():
    # load data
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

    pca = PCA(n_components=.95)
    pca.fit(X_train)

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    svm_errs = []
    with tqdm(desc="Problem 1", total=len(C_VALS)) as pbar:
        for C in C_VALS:
            svm = SVC(C=C, kernel='linear', decision_function_shape='ovo')
            svm.fit(X_train, y_train)
            pbar.update(1)

            svm_errs.append(1 - svm.score(X_test, y_test))

    lr = OVO(LR(solver='lbfgs', max_iter=5000))
    lr.fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)
    err_plot([svm_errs], ["SVM"],
             lr=1. - lr_score,
             title="One vs. One Linear SVM",
             out='hw7/ovo_linear_svm.pdf')

    ovo_svm_errs = []
    with tqdm(desc="Problem 2", total=len(C_VALS)) as pbar:
        for C in C_VALS:
            svm = OVO(SVC(C=C, kernel='poly', degree=3, gamma='auto'))
            svm.fit(X_train, y_train)
            pbar.update(1)

            ovo_svm_errs.append(1 - svm.score(X_test, y_test))

    err_plot([ovo_svm_errs], ["OvO SVM"],
             lr=1. - lr_score,
             title="One vs. One Cubic SVM",
             out='hw7/ovo_cubic_svm.pdf')

    ovr_svm_errs = []
    with tqdm(desc="Problem 3", total=len(C_VALS)) as pbar:
        for C in C_VALS:
            svm = OVR(SVC(C=C, kernel='poly', degree=3, gamma='auto'))
            svm.fit(X_train, y_train)
            pbar.update(1)

            ovr_svm_errs.append(1 - svm.score(X_test, y_test))

    err_plot([ovo_svm_errs, ovr_svm_errs], ["OvO SVM", "OvR SVM"],
             lr=1. - lr_score,
             title="One vs. Rest Cubic SVM/OvO Cubic",
             out='hw7/ovr_cubic_svm.pdf')

    n = 5
    # ensuring that we have at least n neighbors for all classes in the
    # sample
    while True:
        index = np.random.choice(X_train.shape[0], 100, replace=False)

        X_sample = X_train[index]
        y_sample = y_train[index]

        # can use a list comprehension to check
        if all([len(X_sample[y_sample == y_i]) >= n
                for y_i in np.unique(y_sample)]):
            break

    dists = []
    for X_i, y_i in zip(X_sample, y_sample):
        X_cls = X_sample[y_sample == y_i]
        nbrs = NearestNeighbors(n_neighbors=n)
        nbrs.fit(X_cls)
        try:
            distances, _ = nbrs.kneighbors(X_i.reshape(1, -1))
        except ValueError as err:
            raise err
        # nee to use reshape b/c single sample
        dists.append(distances[-1])

    global SIGMA
    SIGMA = np.mean(dists)

    ovo_gauss_svm_errs = []
    with tqdm(desc="Problem 4 (SVM)", total=len(C_VALS),
              file=sys.stdout) as pbar:
        for C in C_VALS:
            svm = OVO(SVC(C=C, kernel='rbf', gamma=1. / (2. * SIGMA ** 2)))
#            svm = SVC(C=C, kernel='rbf', gamma=1. / (2. * SIGMA ** 2),
#                      decision_function_shape='ovo')
            svm.fit(X_train, y_train)
            score = svm.score(X_test, y_test)
            pbar.update(1)

            ovo_gauss_svm_errs.append(1 - score)

    knn_errs = []
    with tqdm(desc="Problem 4 (kNN)", total=len(np.arange(3, 11)),
              file=sys.stdout) as pbar:
        for k in np.arange(3, 11):
            knn = KNeighborsClassifier(n_neighbors=k, weights=gaussian)
            knn.fit(X_train, y_train)
            pbar.update(1)

            knn_errs.append((k, 1 - knn.score(X_test, y_test)))

    err_plot([ovo_gauss_svm_errs], ["OvO SVM"],
             knn=knn_errs,
             title="One vs. One Gaussian SVM with kNN",
             out='hw7/ovo_gaussian_svm_knn.pdf')

    ovr_gauss_svm_errs = []
    with tqdm(desc="Problem 5", total=len(C_VALS),
              file=sys.stdout) as pbar:
        for C in C_VALS:
            svm = OVR(SVC(C=C, kernel='rbf', gamma=1. / (2. * SIGMA ** 2)))
#            svm = SVC(C=C, kernel='rbf', gamma=1. / (2. * SIGMA ** 2),
#                      decision_function_shape='ovr')
            svm.fit(X_train, y_train)
            score = svm.score(X_test, y_test)
            pbar.update(1)

            ovr_gauss_svm_errs.append(1 - score)

    err_plot([ovr_gauss_svm_errs], ["OvR SVM"],
             knn=knn_errs,
             title="One vs. Rest Gaussian SVM with kNN",
             out='hw7/ovr_gaussian_svm_knn.pdf')

    err_plot([svm_errs, ovo_svm_errs, ovr_svm_errs,
              ovo_gauss_svm_errs, ovr_gauss_svm_errs],
             ["Linear SVM", "OvO Cubic SVM", "OvR Cubic SVM",
              "OvO Gaussian SVM", "OvR Gaussian SVM"],
             lr=1. - lr_score,
             knn=knn_errs,
             title="Multiclass SVM Kernels",
             out='hw7/all_svm_knn.pdf')

    min_idx = np.argmin(svm_errs)
    min_lin_err = svm_errs[min_idx]
    min_lin_c = np.log2(C_VALS[min_idx])
    print("Min Linear SVM Error = {0:.4f}".format(min_lin_err))
    print("Min Linear SVM log2(C) = {0}".format(min_lin_c))
    print("LR Error = {0:.4f}".format(1. - lr_score))

    min_idx = np.argmin(ovo_svm_errs)
    min_lin_err = ovo_svm_errs[min_idx]
    min_lin_c = np.log2(C_VALS[min_idx])
    print("Min OvO Cubic SVM Error = {0:.4f}".format(min_lin_err))
    print("Min OvO Cubic SVM log2(C) = {0}".format(min_lin_c))

    min_idx = np.argmin(ovr_svm_errs)
    min_lin_err = ovr_svm_errs[min_idx]
    min_lin_c = np.log2(C_VALS[min_idx])
    print("Min OvR Cubic SVM Error = {0:.4f}".format(min_lin_err))
    print("Min OvR Cubic SVM log2(C) = {0}".format(min_lin_c))

    min_idx = np.argmin(knn_errs)
    min_lin_k, min_lin_err = knn_errs[min_idx]
    print("Min kNN Error = {0:.4f}".format(min_lin_err))
    print("Min kNN log2(C) = {0}".format(min_lin_k))

    min_idx = np.argmin(ovo_gauss_svm_errs)
    min_lin_err = ovo_gauss_svm_errs[min_idx]
    min_lin_c = np.log2(C_VALS[min_idx])
    print("Min OvO Gaussian SVM Error = {0:.4f}".format(min_lin_err))
    print("Min OvO Gaussian SVM log2(C) = {0}".format(min_lin_c))

    min_idx = np.argmin(ovr_gauss_svm_errs)
    min_lin_err = ovr_gauss_svm_errs[min_idx]
    min_lin_c = np.log2(C_VALS[min_idx])
    print("Min OvR Gaussian SVM Error = {0:.4f}".format(min_lin_err))
    print("Min OvR Gaussian SVM log2(C) = {0}".format(min_lin_c))

    print("sigma = {0:.4f}".format(SIGMA))


if __name__ == '__main__':
    figure = {'figsize': (12, 12)}
    mpl.rc('figure', **figure)

    font = {'size': 24}
    mpl.rc('font', **font)

    directory = os.path.join(os.path.dirname(__file__), 'hw7')
    if not os.path.exists(directory):
        os.makedirs(directory)
#    plt.xkcd()
    main()
