# -*- coding: utf-8 -*-
"""
 Initialization of the training sets for the NN function.
 WARNNG : running this file will create a new neural network and therefore
 change the existing parameters. Only run this file when you want to reset the
 nn parameters
"""

import sys
from preprocess import np, pkl, create_data
# Pour Vivi :
sys.path.insert(1, "C:/Users/viniv/OneDrive/Bureau/MOPSI2/")
from network import *

# Number of points on which the value of the function is known.
N = 2000

# Build a neural network with 2 layers of width 3
nn = Net2(3)

def n(x: float):
    """ Function that is calculated thanks to a neural network
    
    Arguments:
        x {float}
    
    Returns:
        float
    """
    x_tensor = torch.FloatTensor([x])
    y_tensor = nn(x_tensor)
    y = y_tensor.item()
    return y


create_data(N, n, "nnW3L2")
path = "C:/Users/viniv/OneDrive/Bureau/MOPSI2/preprocessing/nnw3L2.py"
torch.save(nn.state_dict(), path)