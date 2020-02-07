""" Initialization of the training sets of the hat function.
"""
# Pour Louise et Vivi
from preprocess import np, pkl, create_data

# Pour Jean :
# from preprocess import np, pkl, create_data

# Number of points on which the value of the function is known.
N = 2000


def g(x: float):
    """ The hat function that will be used to check
    the quality of the approximation.

    Arguments:
        x {float} -- Will be taken in [0, 1]

    Returns:
        float
    """
    res = 2 * x
    if x > 0.5:
        res = 2 * (1 - x)
    return res


create_data(N, g, "hat")
