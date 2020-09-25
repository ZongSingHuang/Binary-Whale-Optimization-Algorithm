# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 08:27:39 2020

@author: ZongSing_NB
"""

from BWOA import BWOA
import numpy as np
import time
import pandas as pd
import functools
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

wine = datasets.load_wine()
breast = datasets.load_breast_cancer()
waveform = np.loadtxt("waveform-5000_csv.csv", skiprows=1, delimiter=",")

def fitness(x, X_train, y_train, X_valid, y_valid, evalute=False):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    loss = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        feature = x[i, :] == 1
        if sum(feature)>0:
            model = SVC(kernel='rbf',  decision_function_shape='ovo').fit(X_train[:, feature], y_train)
            acc_train = accuracy_score(y_train, model.predict(X_train[:, feature]))
            acc_valid = accuracy_score(y_valid, model.predict(X_valid[:, feature]))
            if evalute:
                loss[i] = acc_valid
            else:
                loss[i] = 0.75*(1-acc_valid) + 0.25*(sum(feature)/x.shape[1])
        else:
            loss[i] = np.inf
    
    return loss


max_iter = 70
num_particle = 8
times = 1
table = np.zeros((6, 23))
table[2, :] = np.ones(23)*np.inf
table[3, :] = -np.ones(23)*np.inf
ALL = np.zeros((times, 23))
for i in range(times):
    X = wine.data
    y = wine.target
    num_dim = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_valid = sc.transform(X_valid)
    X_test = sc.transform(X_test)
    fit_func = functools.partial(fitness, 
                                 X_train=X_train, y_train=y_train, 
                                 X_valid=X_valid, y_valid=y_valid)
    optimizer = BWOA(fit_func=fit_func, num_dim=num_dim, num_particle=num_particle, max_iter=max_iter)
    start = time.time()
    optimizer.opt()
    end = time.time()
    score = fitness(optimizer.gBest_X, 
                    np.vstack((X_train, X_valid)), np.hstack((y_train, y_valid)), 
                    X_test, y_test, evalute=True)
    if score==np.inf or score==-np.inf:
        print(456)
    if score<table[2, 0]: table[2, 0] = score
    if score>table[3, 0]: table[3, 0] = score
    table[0, 0] += score
    table[1, 0] += end - start
    table[5, 0] +=sum(optimizer.gBest_X)
    ALL[i, 0] = score
    
    X = breast.data
    y = breast.target
    num_dim = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_valid = sc.transform(X_valid)
    X_test = sc.transform(X_test)
    fit_func = functools.partial(fitness, 
                                 X_train=X_train, y_train=y_train, 
                                 X_valid=X_valid, y_valid=y_valid)
    optimizer = BWOA(fit_func=fit_func, num_dim=num_dim, num_particle=num_particle, max_iter=max_iter)
    start = time.time()
    optimizer.opt()
    end = time.time()
    score = fitness(optimizer.gBest_X, 
                    np.vstack((X_train, X_valid)), np.hstack((y_train, y_valid)), 
                    X_test, y_test, evalute=True)
    if score==np.inf or score==-np.inf:
        print(789)
    if score<table[2, 1]: table[2, 1] = score
    if score>table[3, 1]: table[3, 1] = score
    table[0, 1] += score
    table[1, 1] += end - start
    table[5, 1] +=sum(optimizer.gBest_X)
    ALL[i, 1] = score

    # Too Slow!!!!
    X = waveform[:, :-1]
    y = waveform[:, -1]
    num_dim = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_valid = sc.transform(X_valid)
    X_test = sc.transform(X_test)
    fit_func = functools.partial(fitness, 
                                  X_train=X_train, y_train=y_train, 
                                  X_valid=X_valid, y_valid=y_valid)
    optimizer = BWOA(fit_func=fit_func, num_dim=num_dim, num_particle=num_particle, max_iter=max_iter)
    start = time.time()
    optimizer.opt()
    end = time.time()
    score = fitness(optimizer.gBest_X, 
                    np.vstack((X_train, X_valid)), np.hstack((y_train, y_valid)), 
                    X_test, y_test, evalute=True)
    if score==np.inf or score==-np.inf:
        print(789)
    if score<table[2, 2]: table[2, 2] = score
    if score>table[3, 2]: table[3, 2] = score
    table[0, 2] += score
    table[1, 2] += end - start
    table[5, 2] +=sum(optimizer.gBest_X)
    ALL[i, 2] = score
     
    print(i+1)
    
    
table[:2, :] = table[:2, :] / times
table[5, :] = table[5, :] / times
table[4, :] = np.std(ALL, axis=0)
table = pd.DataFrame(table)
table.columns=['Wine_BWOA', 'Breast_BWOA', 'Waveform_BWOA', 'F4', 'F5', 'F6', 'F7', 
                'F8', 'F9', 'F10', 'F11', 'F12', 
                'F13', 'F14', 'F15', 'F16', 'F17', 'F18',
                'F19', 'F20', 'F21', 'F22', 'F23']
table.index = ['avg', 'time', 'worst', 'best', 'std', 'feature']