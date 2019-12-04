""" Enables to call the general function for training.
"""
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI")
from training import training
from constant import *


training(W, L, N, NB_EPOCH, learning_rate, id_function)
