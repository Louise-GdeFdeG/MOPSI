""" This files contains all the variables we are using for each tests.
This enables not to have to change the variables value everywhere."""

"""
W = [5, 10, 50, 100]
L = [2, 3, 4, 5, 7, 10]
N = [100, 500, 1000, 2000]
NB_EPOCH = [1, 5, 10, 50, 100]
Learning_rate = [1e-4, 1e-3, 1e-2]"""

# Width of the nn
W = 7
# Widths we will use
WW = [5, 10, 50, 100]
# Depth of the nn (number of layer)
L = 7
# Depths we will use
LL = [2, 3, 4, 5, 7, 10]

# The number of known points of the function.
N = 2000
# Number of epoch = number of time our nn trains on the data.
# (it understands fast but we have to explain to it several time)
NB_EPOCH = 20
# The maximum gap we want between the target value and the value given
# by the model
learning_rate = 0.001

# Number of times we will calculate the error for a given (W, L)
N_ite = 5

# The identification of the function
id_function = "piecewise3"

# For constant function : id_function = "constant"
# For piecewise function : id_function = "piecewise" + number (2,4,10)
# For polyomial function : id_function = "polynomial"
# For hat function : id_function = "hat"
# For truncated hat function : id_function = "truncated_hat"
