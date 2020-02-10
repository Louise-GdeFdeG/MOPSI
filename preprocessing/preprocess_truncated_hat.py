""" Initialization of the training sets of the hat function.
"""
# Pour Louise et Vivi
from preprocess import np, pkl

# Pour Jean :
# from preprocess import np, pkl, create_data

# Number of points on which the value of the function is known.
N = 2000

# Gap with for the hat truncated function :
gap_width = 0.7


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


def create_data(N: int, f, function: str):
    """This function is used to create the sets of data. It also saves them
    following a particular syntax for the file names.

    Arguments:
        N {int} -- The number of known points of the function.
        f {function} -- The function we are interested in.
        function {str} -- The function identification.
    """

    # Creation of the training sets
    list_abscissa = np.linspace(0, 1, N)
    list_abscissa_left = np.linspace(0, 0.5-gap_width/2., N // 2)
    list_abscissa_right = np.linspace(0.5+gap_width/2., 1, N // 2)
    data_f_training = [(x, f(x)) for x in list_abscissa_left] + [(x, f(x)) for x in list_abscissa_right]
    data_f = [(x, f(x)) for x in list_abscissa]
    length = len(list_abscissa_left) + len(list_abscissa_right)
    
    # Desorganise the sets
    np.random.shuffle(data_f_training)
    np.random.shuffle(data_f)

    # Spliting of the sets : 80% used for training, 10% for validation
    # and 10% for testing
    index_80pc = int(length * 0.8)
    index_90pc = int(length * 0.9)
    training_set_f, validation_set_f, test_set_f = (
        data_f_training[:index_80pc],
        data_f[index_80pc:index_90pc],
        data_f[index_90pc:],
    )

    # We save the sets
    dictionnary_f = {
        "train": training_set_f,
        "valid": validation_set_f,
        "test": test_set_f,
    }
    data_file = "data_" + function + "_" + str(N) + ".pkl"
    # Pour Louise :
    #path = "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI/preprocessing/"

    # Pour Vivi : "C:/Users/viniv/OneDrive/Bureau/MOPSI/MOPSI/preprocessing/"
    # path = "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI/preprocessing/"
    # Pour Vivi :
    path = "C:/Users/viniv/OneDrive/Bureau/MOPSI2/preprocessing/"
    # Pour Jean :
    # path = "/Users/Jean/Documents/Ponts/MOPSI/MOPSI/preprocessing/"
    with open(path + data_file, "wb") as f:
        pkl.dump(dictionnary_f, f)


create_data(N, g, "truncated_hat")
