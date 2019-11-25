# -*- coding: utf-8 -*-


""" Initialization of the training sets.
"""
import numpy as np
import pickle as pkl

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


# Creation of the training sets
list_abscissa = np.linspace(0, 1, N)
data_f = [(x, f(x)) for x in list_abscissa]
np.random.shuffle(data_f)

# Spliting of the sets : 80% used for training, 10% for validation
# and 10% for testing
index_80pc = int(len(list_abscissa) * 0.8)
index_90pc = int(len(list_abscissa) * 0.9)
training_set_f, validation_set_f, test_set_f = (
    data_f[:index_80pc],
    data_f[index_80pc:index_90pc],
    data_f[index_90pc:],
)
# We save the sets
dictionary_f = {"train": training_set_f, "valid": validation_set_f, "test": test_set_f}
with open("data_f.pkl", "wb") as file:
    pkl.dump(dictionary_f, file)
