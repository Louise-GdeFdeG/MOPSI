""" Enables to call the general function for testing.
"""
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI")
from testing import testing

N = 2000

testing(N, "constant")
