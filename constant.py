""" This files contains all the variables we are using for each tests.
This enables not to have to change the variables value everywhere."""

"""
W = [5, 10, 50, 100]
L = [2, 5, 7, 10]
N = [100, 500, 1000, 2000]
NB_EPOCH = [1, 5, 10, 50, 100]
Learning_rate = [1e-4, 1e-3, 1e-2]"""

# Width of the nn
W = 5
# Depth of the nn (number of layer)
L = 2
# The number of known points of the function.
N = 2000
# Number of epoch = number of time our nn trains on the data.
# (it understands fast but we have to explain to it several time)
NB_EPOCH = 100
# The maximum gap we want between the target value and the value given
# by the model
learning_rate = 0.0001

# The identification of the function
id_function = "piecewise"
# For constant function : id_function = "constant"
# For piecewise function : id_function = "piecewise"
# For polyomial function : id_function = "polynomial"

