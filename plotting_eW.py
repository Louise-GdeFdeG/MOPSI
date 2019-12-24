# -*- coding: utf-8 -*-
"""
Plot the total error for L constant, and W taking values chosen in constant.py
"""
from testing import testing
from training import training
from constant import WW, L, N, N_ite, learning_rate, NB_EPOCH, id_function
import numpy as np
import matplotlib.pyplot as plt

errormax = []
errorplot = []
errormin = []
for wite in WW:
    errorw = []
    for n in range(N_ite):
        training(wite, L, N, NB_EPOCH, learning_rate, id_function)
        errorw.append(testing(wite, L, N, NB_EPOCH, learning_rate,
                              id_function))
    means = np.mean(errorw)
    sigma = np.sqrt(np.var(errorw, ddof=1))
    errormax.append(means + sigma)
    errorplot.append(means)
    errormin.append(means - sigma)

plt.clf()
plt.figure()

fig, ax1 = plt.subplots()

ax1.set_xlabel("W", fontsize=14)
ax1.set_ylabel("Error", fontsize=14)
ax1.plot(WW, errormax, "b+", label="Error + Sigma")
#Errormax and errormin won't plot if N_ite ==1
ax1.plot(WW, errorplot, "r+", label="Average error")
ax1.plot(WW, errormin, "g+", label="Error - Sigma")

plt.title("Total error on approximation of "+id_function+" depending on W with constants L=" + str(L) + " NB_EPOCH=" + str(NB_EPOCH),
          fontsize=16)
plt.legend(loc="upper right")
plt.show()
