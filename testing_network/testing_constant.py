""" Enables to call the general function for testing.
"""
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI")
from testing import testing
from constant import *


testing(W, L, N, NB_EPOCH, learning_rate, id_function)
