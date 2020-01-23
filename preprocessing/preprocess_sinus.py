# -*- coding: utf-8 -*-
"""
Sinus function
"""
from preprocessing.preprocess import np, pkl, create_data
import math

# Number of points on which the value of the function is known.
N = 2000


def j(x: float):
    """The C_infinite function that will be used to check the quality
    of the approximation. Roots : 1 and 0.5

    Arguments:
        x {float} - Will be taken in [0, 1]

    Returns:
        float
    """
    return math.sin(2*math.pi*x)


create_data(N, j, "sinus")
