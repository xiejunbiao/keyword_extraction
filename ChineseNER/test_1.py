# -*- coding: utf-8 -*-
"""
Create Time: 2020/8/20 16:51
Author: xiejunbiao
"""
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score


def t2():
    X = np.array([[1, 2], [13, 4], [1, 20], [32, 41], [17, 28], [130, 48], [91, 270], [302, 410]])
    y = np.array([1, 2, 3, 4, 1, 2, 3, 4])

    kf = KFold(n_splits=5)
    for i in kf.split(X):
        print(i)
    for train_index, test_index in kf.split(X):
        # print('train_index', train_index, 'test_index', test_index)
        train_X, train_y = X[train_index], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        print('tr', train_X, 'te',  train_y)
        print(test_X, test_y)


def t1():
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4])

    kf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=0)
    for i in kf.split(X):
        print(i)
    for train_index, test_index in kf.split(X):
        print('train_index', train_index, 'test_index', test_index)


def train_():
    iris = datasets.load_iris()

    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)

    print(scores)


if __name__ == '__main__':
    t2()

