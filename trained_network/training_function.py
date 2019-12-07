""" Enables to call the general function for training.
"""
import sys

# insert at 1, 0 is the script path (or '' in REPL)
# Pour Louise :
sys.path.insert(1, "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI")
# Pour Vivi :
sys.path.insert(1, "C:/Users/viniv/OneDrive/Bureau/MOPSI/MOPSI/")
from training import training
from constant import *


training(W, L, N, NB_EPOCH, learning_rate, id_function)
