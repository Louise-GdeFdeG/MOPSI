""" Definition of our neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net2(nn.Module):
    """ The class that defines a network with 2 layers. It's a subclass
    of the class Module from torch.nn.
    """

    def __init__(self, W: int):
        super(Net2, self).__init__()
        # an affine operation (here we have 2 layers): y = Wx + b
        self.lin_1 = nn.Linear(1, W)  # the input of the first layer is a float
        self.lin_2 = nn.Linear(W, 1)  # the output of the last layer is a float

    def forward(self, x: float):
        """ Describes how our model works. The input passes through each
        linear layer lin_i, between each layer the relu function is applied.

        Arguments:
            x {float} -- The abscissa value x.

        Returns:
            float -- The approximation of our model of f(x)
        """
        out_lin_1 = F.relu(self.lin_1(x))
        out_lin_2 = self.lin_2(out_lin_1)
        return out_lin_2


class Net3(nn.Module):
    """ The class that defines a network with 3 layers. It's a subclass
    of the class Module from torch.nn.
    """

    def __init__(self, W: int):
        super(Net3, self).__init__()
        # an affine operation (here we have 3 layers): y = Wx + b
        self.lin_1 = nn.Linear(1, W)  # the input of the first layer is a float
        self.lin_2 = nn.Linear(W, W)
        self.lin_3 = nn.Linear(W, 1)  # the output of the last layer is a float

    def forward(self, x: float):
        """ Describes how our model works. The input passes through each
        linear layer lin_i, between each layer the relu function is applied.

        Arguments:
            x {float} -- The abscissa value x.

        Returns:
            float -- The approximation of our model of f(x)
        """
        out_lin_1 = F.relu(self.lin_1(x))
        out_lin_2 = F.relu(self.lin_2(out_lin_1))
        out_lin_3 = self.lin_3(out_lin_2)
        return out_lin_3


class Net4(nn.Module):
    """ The class that defines a network with 4 layers. It's a subclass
    of the class Module from torch.nn.
    """

    def __init__(self, W: int):
        super(Net4, self).__init__()
        # an affine operation (here we have 3 layers): y = Wx + b
        self.lin_1 = nn.Linear(1, W)  # the input of the first layer is a float
        self.lin_2 = nn.Linear(W, W)
        self.lin_3 = nn.Linear(W, W)
        self.lin_4 = nn.Linear(W, 1)  # the output of the last layer is a float

    def forward(self, x: float):
        """ Describes how our model works. The input passes through each
        linear layer lin_i, between each layer the relu function is applied.

        Arguments:
            x {float} -- The abscissa value x.

        Returns:
            float -- The approximation of our model of f(x)
        """
        out_lin_1 = F.relu(self.lin_1(x))
        out_lin_2 = F.relu(self.lin_2(out_lin_1))
        out_lin_3 = F.relu(self.lin_3(out_lin_2))
        out_lin_4 = self.lin_4(out_lin_3)
        return out_lin_4


class Net5(nn.Module):
    """ The class that defines a network with 5 layers. It's a subclass
    of the class Module from torch.nn.
    """

    def __init__(self, W: int):
        super(Net5, self).__init__()
        # an affine operation (here we have 3 layers): y = Wx + b
        self.lin_1 = nn.Linear(1, W)  # the input of the first layer is a float
        self.lin_2 = nn.Linear(W, W)
        self.lin_3 = nn.Linear(W, W)
        self.lin_4 = nn.Linear(W, W)
        self.lin_5 = nn.Linear(W, 1)  # the output of the last layer is a float

    def forward(self, x: float):
        """ Describes how our model works. The input passes through each
        linear layer lin_i, between each layer the relu function is applied.

        Arguments:
            x {float} -- The abscissa value x.

        Returns:
            float -- The approximation of our model of f(x)
        """
        out_lin_1 = F.relu(self.lin_1(x))
        out_lin_2 = F.relu(self.lin_2(out_lin_1))
        out_lin_3 = F.relu(self.lin_3(out_lin_2))
        out_lin_4 = F.relu(self.lin_4(out_lin_3))
        out_lin_5 = self.lin_5(out_lin_4)
        return out_lin_5


class Net7(nn.Module):
    """ The class that defines a network with 7 layers. It's a subclass
    of the class Module from torch.nn.
    """

    def __init__(self, W: int):
        super(Net7, self).__init__()
        # an affine operation (here we have 3 layers): y = Wx + b
        self.lin_1 = nn.Linear(1, W)  # the input of the first layer is a float
        self.lin_2 = nn.Linear(W, W)
        self.lin_3 = nn.Linear(W, W)
        self.lin_4 = nn.Linear(W, W)
        self.lin_5 = nn.Linear(W, W)
        self.lin_6 = nn.Linear(W, W)
        self.lin_7 = nn.Linear(W, 1)  # the output of the last layer is a float

    def forward(self, x: float):
        """ Describes how our model works. The input passes through each
        linear layer lin_i, between each layer the relu function is applied.

        Arguments:
            x {float} -- The abscissa value x.

        Returns:
            float -- The approximation of our model of f(x)
        """
        out_lin_1 = F.relu(self.lin_1(x))
        out_lin_2 = F.relu(self.lin_2(out_lin_1))
        out_lin_3 = F.relu(self.lin_3(out_lin_2))
        out_lin_4 = F.relu(self.lin_4(out_lin_3))
        out_lin_5 = F.relu(self.lin_5(out_lin_4))
        out_lin_6 = F.relu(self.lin_6(out_lin_5))
        out_lin_7 = self.lin_7(out_lin_6)
        return out_lin_7


class Net10(nn.Module):
    """ The class that defines a network with 10 layers. It's a subclass
    of the class Module from torch.nn.
    """

    def __init__(self, W: int):
        super(Net10, self).__init__()
        # an affine operation (here we have 3 layers): y = Wx + b
        self.lin_1 = nn.Linear(1, W)  # the input of the first layer is a float
        self.lin_2 = nn.Linear(W, W)
        self.lin_3 = nn.Linear(W, W)
        self.lin_4 = nn.Linear(W, W)
        self.lin_5 = nn.Linear(W, W)
        self.lin_6 = nn.Linear(W, W)
        self.lin_7 = nn.Linear(W, W)
        self.lin_8 = nn.Linear(W, W)
        self.lin_9 = nn.Linear(W, W)
        self.lin_10 = nn.Linear(W, 1)  # the output oflast layer is a float

    def forward(self, x: float):
        """ Describes how our model works. The input passes through each
        linear layer lin_i, between each layer the relu function is applied.

        Arguments:
            x {float} -- The abscissa value x.

        Returns:
            float -- The approximation of our model of f(x)
        """
        out_lin_1 = F.relu(self.lin_1(x))
        out_lin_2 = F.relu(self.lin_2(out_lin_1))
        out_lin_3 = F.relu(self.lin_3(out_lin_2))
        out_lin_4 = F.relu(self.lin_4(out_lin_3))
        out_lin_5 = F.relu(self.lin_5(out_lin_4))
        out_lin_6 = F.relu(self.lin_6(out_lin_5))
        out_lin_7 = F.relu(self.lin_7(out_lin_6))
        out_lin_8 = F.relu(self.lin_8(out_lin_7))
        out_lin_9 = F.relu(self.lin_9(out_lin_8))
        out_lin_10 = self.lin_10(out_lin_9)
        return out_lin_10

