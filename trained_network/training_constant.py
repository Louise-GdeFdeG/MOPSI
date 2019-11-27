""" Enables to call the general function for training.
"""
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI")
from training import training

N = 2000

training(N, "constant")
