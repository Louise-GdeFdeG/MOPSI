""" Initialization of the training sets of the polynomial function.
"""
from preprocessing.preprocess import np, pkl, create_data

# Number of points on which the value of the function is known.
N = 2000


def f(x: float):
    """The C_infinite function that will be used to check the quality
    of the approximation. Roots : 1 and 0.5

    Arguments:
        x {float} - Will be taken in [0, 1]

    Returns:
        float
    """
    return 2 * x * x - 3 * x + 1


create_data(N, f, "polynomial")
