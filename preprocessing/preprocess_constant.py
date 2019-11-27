""" Initialization of the training sets for the constant function.
"""
from preprocessing.preprocess import np, pkl, create_data

# Number of points on which the value of the function is known.
N = 2000


def h(x: float):
    """ Constant function equal to 1.
    
    Arguments:
        x {float}
    
    Returns:
        float -- 1
    """
    return 1.0


create_data(N, h, "constant")
