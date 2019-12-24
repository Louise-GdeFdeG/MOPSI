""" Enables to call the general function for testing.
"""
import sys

# insert at 1, 0 is the script path (or '' in REPL)
# Pour Louise:
#sys.path.insert(1, "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI")
# Pour Vivi :
sys.path.insert(1, "C:/Users/viniv/OneDrive/Bureau/MOPSI2/")
from testing import testing
from constant import *


testing(W, L, N, NB_EPOCH, learning_rate, id_function)
