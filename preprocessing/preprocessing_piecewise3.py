# -*- coding: utf-8 -*-
"""
 Initialization of the training set for a continuous piecewise function
 with 2 discontinuity points
"""

# Pour Louise et Vivi
# from preprocessing.preprocess import np, pkl, create_data
# Pour Jean : 
from preprocess import np, pkl, create_data

# Number of points on which the value of the function is known.
N = 2000


def g3(x: float):
    """ The piecewise continuous function that will be used to check
    the quality of the approximation.

    Arguments:
        x {float} -- Will be taken in [0, 1]

    Returns:
        int -- 0 or 1 or 2
    """
    return (int(x * 3)/2.)


create_data(N, g3, "piecewise3")