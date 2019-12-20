# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 18:59:06 2019

@author: viniv
"""
from testing import * 
from training import *
from constant import * 
import statistics
import numpy as np
import matplotlib.pyplot as plt

errormax = []
errorplot = []
errormin = []
for wite in WW:
    errorw = []
    for n in range(N_ite):    
        training(wite, L, N, NB_EPOCH, learning_rate, id_function)
        errorw.append(testing(wite, L, N, NB_EPOCH, learning_rate, id_function))
    means = np.mean(errorw)
    sigma = np.sqrt(np.var(errorw, ddof = 1))
    errormax.append(means + sigma)
    errorplot.append(means)
    errormin.append(means - sigma)

plt.clf()
plt.figure()

fig, ax1 = plt.subplots()

ax1.set_xlabel("W", fontsize=14)
ax1.set_ylabel("Error", fontsize=14)
ax1.plot(WW, errormax, "b+", label="Error + Sigma")
ax1.plot(WW, errorplot, "r+", label="Average error")
ax1.plot(WW, errormin, "g+", label = "Error - Sigma")

plt.legend(loc="upper right")
plt.show()

