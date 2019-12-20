""" Initialization of the training sets ot the piecewise continuous function.
"""
# Pour Louise et Vivi
# from preprocessing.preprocess import np, pkl, create_data
# Pour Jean : 
from preprocess import np, pkl, create_data

# Number of points on which the value of the function is known.
N = 2000


def g(x: float):
    """ The piecewise continuous function that will be used to check
    the quality of the approximation.

    Arguments:
        x {float} -- Will be taken in [0, 1]

    Returns:
        int -- the digit of the first decimal of x.
    """
    return int(x * 10)


create_data(N, g, "piecewise")
