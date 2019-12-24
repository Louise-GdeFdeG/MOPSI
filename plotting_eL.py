# -*- coding: utf-8 -*-
"""
Plots the total error for L constant, and W taking values chosen in constant.py
"""

from testing import testing
from training import training
from constant import W, LL, N, N_ite, learning_rate, NB_EPOCH, id_function
import numpy as np
import matplotlib.pyplot as plt

errormax = []
errorplot = []
errormin = []
for lite in LL:
    errorl = []
    for n in range(N_ite):
        training(W, lite, N, NB_EPOCH, learning_rate, id_function)
        errorl.append(testing(W, lite, N, NB_EPOCH, learning_rate, id_function))
    means = np.mean(errorl)
    sigma = np.sqrt(np.var(errorl, ddof=1))
    errormax.append(means + sigma)
    errorplot.append(means)
    errormin.append(means - sigma)

plt.clf()
plt.figure()

fig, ax1 = plt.subplots()

ax1.set_xlabel("L", fontsize=14)
ax1.set_ylabel("Error", fontsize=14)
ax1.plot(LL, errormax, "b+", label="Error + Sigma")
ax1.plot(LL, errorplot, "r+", label="Average error")
ax1.plot(LL, errormin, "g+", label="Error - Sigma")

plt.title("Total error on approximation of "+id_function+" depending on L with constants W=" + str(W) + " NB_EPOCH=" + str(NB_EPOCH) + " N_ite="+str(N_ite),
          fontsize=16)
plt.legend(loc="upper right")
plt.show()
