# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 08:27:39 2020

@author: ZongSing_NB
"""

from BWOA import BWOA
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

# 讀資料
Zoo = pd.read_csv('Zoo.csv', header=None).values

X_train, X_test, y_train, y_test = train_test_split(Zoo[:, :-1], Zoo[:, -1], stratify=Zoo[:, -1], test_size=0.5)

def Zoo_test(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    loss = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        if np.sum(x[i, :])>0:
            knn = KNeighborsClassifier(n_neighbors=5).fit(X_train[:, x[i, :]], y_train)
            score = accuracy_score(knn.predict(X_test[:, x[i, :]]), y_test)
            loss[i] = 0.99*(1-score) + 0.01*(np.sum(x[i, :])/X_train.shape[1])
        else:
            loss[i] = np.inf
            print(666)
    return loss

optimizer = BWOA(fit_func=Zoo_test, 
                 num_dim=X_train.shape[1], num_particle=5, max_iter=70, x_max=1, x_min=0)
optimizer.opt()

feature = Zoo[:, :-1]
label = Zoo[:, -1]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[:, optimizer.gBest_X], y_train)
print(accuracy_score(knn.predict(X_test[:, optimizer.gBest_X]), y_test))