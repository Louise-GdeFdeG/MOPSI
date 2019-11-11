""" Definition of our neural network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """ The class that defines our modele. It's a subclass
    of the class Module from torch.nn.
    """

    def __init__(self, W: int):
        super(Net, self).__init__()
        # an affine operation (here we have 3 layers): y = Wx + b
        self.lin_1 = nn.Linear(1, W)  # the input of the first layer is a float
        self.lin_2 = nn.Linear(W, W)
        self.lin_3 = nn.Linear(W, 1)  # the output of the last layer is a float

    def forward(self, x: float):
        """ Describes how our model works. The input passes through each
        linear layer lin_i, and between each layer the relu function is applied.
        
        Arguments:
            x {float} -- The abscissa value x.
        
        Returns:
            float -- The approximation of our model of f(x)
        """
        out_lin_1 = F.relu(self.lin_1(x))
        out_lin_2 = F.relu(self.lin_2(out_lin_1))
        return F.relu(self.lin_3(out_lin_2))
