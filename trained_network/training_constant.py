""" Enables to call the general function for training.
"""
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI")
from training import training

N = 2000

"""The constants used, the ones we are going to change to assess
their impact on the quality on the model."""
# Width of the nn
W = 5
# Depth of the nn (number of layer)
L = 2
# Number of epoch = number of time our nn trains on the data.
# (it understands fast but we have to explain to it several time)
NB_EPOCH = 10
# The maximum gap we want between the target value and the value given
# by the model
learning_rate = 0.0001

training(W, L, 2000, NB_EPOCH, learning_rate, "constant")
