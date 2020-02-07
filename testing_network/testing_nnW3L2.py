# -*- coding: utf-8 -*-
"""
In this file we test the network that learned and check if it has the same 
parameters as the network given in input : nnW3L2
"""

import sys

# insert at 1, 0 is the script path (or '' in REPL)
# Pour Louise:
#sys.path.insert(1, "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI")
# Pour Vivi :
sys.path.insert(1, "C:/Users/viniv/OneDrive/Bureau/MOPSI2/")
from testing import testing
from constant import *
from network import *

testing(W, L, N, NB_EPOCH, learning_rate, id_function)




