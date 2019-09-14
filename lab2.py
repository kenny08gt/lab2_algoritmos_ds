#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 09:23:35 2019

@author: alanhurtarte
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient(x, Q, c):
    mul_q_x = np.matmul(Q, x)
#    print(mul_q_x)
    return mul_q_x + c

def gradientDescent(x0, epsilon, N, Q, c, learning_rate):
    xi = x0
    f_hat = gradient(xi, Q, c)
    i = 0
    epochs = []
    f_vals = []
    directions = []
    Xn = []
    axs = []
    while np.linalg.norm(f_hat) >= epsilon and i < N:
        if hasattr(learning_rate, '__call__'):
            ak = learning_rate()
        else:
            ak = learning_rate
        f_hat = gradient(xi, Q, c)
        xi = xi - ak * f_hat
        i += 1
        epochs.append(i)
        Xn.append(xi)
        directions.append(-1*(f_hat))
        f_vals.append(np.linalg.norm(f_hat))
    
    results = pd.DataFrame({'Iter':epochs, 'Xn':Xn, 'Directions': directions, 'grad_abs': f_vals})
    return results, f_hat
  
    
    
def __main__():
    x = np.array([[3], [5], [7]])
    Q = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    c = np.array([[1], [0], [1]])
    e = 10**-6
    N = 30
    lr = 0.01
    data, f = gradientDescent(x, e, N, Q, c,  lr)
    pd.set_option('display.expand_frame_repr', False)
    print(data)
    plt.plot(data['grad_abs'], data['Iter'])
    plt.ylabel('Iteration (k)')
    plt.xlabel('Gradient norm')
    plt.suptitle('Learning rate '+str(lr))
    plt.show()
#    print(x + c)
    
if __name__ == '__main__':
    __main__()