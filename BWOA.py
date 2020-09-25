# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 08:27:39 2020

@author: ZongSing_NB

Main reference:https://doi.org/10.1007/s13042-019-00996-5
"""

import numpy as np
import matplotlib.pyplot as plt

class BWOA():
    def __init__(self, fit_func, num_dim=30, num_particle=20, max_iter=500):
        self.fit_func = fit_func
        self.num_particle = num_particle
        self.num_dim = num_dim
        self.iter = 0
        self.max_iter = max_iter
        self.X = np.random.choice([0, 1], size=(self.num_particle, self.num_dim))
        self.gBest_X = None
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(self.max_iter)
        self.Lbest = []
        
    def opt(self):
        b = 1
        while(self.iter<self.max_iter):
            a = 2 * ( 1 - self.iter/self.max_iter )
            for i in range(self.num_particle):
                score = self.fit_func(self.X[i])
                
                if score.min()<self.gBest_score:
                    self.gBest_score = score.min().copy()
                    self.gBest_X = self.X[score.argmin()].copy()
                    self.Lbest.append(self.X[score.argmin()].copy())
                
                if self.iter>self.max_iter/3:
                    idx = int( np.random.randint(low=0, high=len(self.Lbest), size=1) )
                    self.gBest_X = self.Lbest[idx].copy()

                r1 = np.random.uniform()
                r2 = np.random.uniform()
                A = 2*a*r1 - a
                C = 2*r2
                self.l = np.random.uniform(low=-1, high=1)
                p = np.random.uniform()               
                
                for j in range(self.num_dim):
                    rd = np.random.uniform()
                    if p<0.4:
                        D = np.abs( C*self.gBest_X[j] - self.X[i,j] )
                        TF = np.abs( np.pi/3 * np.arctan(np.pi/3 * A * D) + 0.02 )
                        if np.abs(A)<1:
                            if rd<TF:
                                self.X[i,j] = 1 - self.X[i,j]
                            else:
                                self.X[i,j] = self.gBest_X[j].copy()
                        else:
                            if rd<TF:
                                self.X[i,j] = 1 - self.X[np.random.randint(low=0, 
                                    high=self.num_particle, size=1),j]
                            else:
                                self.X[i,j] = self.X[np.random.randint(low=0, 
                                    high=self.num_particle, size=1),j].copy()
                    else:
                        D = np.abs( self.gBest_X[j] - self.X[i,j] )
                        S = D*np.exp(b*self.l)*np.cos(2*np.pi*self.l)
                        TF = np.abs( np.arctan(S) + 0.09 )
                        if rd>0.92 and TF==0.09/4:
                            self.X[i,j] = 1 - self.gBest_X[j]
                        elif rd>TF:
                            self.X[i,j] = self.gBest_X[j].copy()
                        else:
                            self.X[i,j] = self.X[i,j].copy()
            
            self.gBest_curve[self.iter] = self.gBest_score.copy()
            self.iter += 1
     
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()        
            