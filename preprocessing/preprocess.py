""" Definition of the general function that is used to create and save de data.
"""
import numpy as np
import pickle as pkl


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
    data_f = [(x, f(x)) for x in list_abscissa]

    # Desorganise the sets
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
    dictionnary_f = {
        "train": training_set_f,
        "valid": validation_set_f,
        "test": test_set_f,
    }
    data_file = "data_" + function + "_" + str(N) + ".pkl"
    # Pour Louise :
    # path = "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI/preprocessing/"
<<<<<<< HEAD
    # Pour Vivi : "C:/Users/viniv/OneDrive/Bureau/MOPSI/MOPSI/preprocessing/"
    path = "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI/preprocessing/"
=======
    # Pour Vivi :
    path = "C:/Users/viniv/OneDrive/Bureau/MOPSI/MOPSI/preprocessing/"
    # Pour Jean : 
    #path = "/Users/Jean/Documents/Ponts/MOPSI/MOPSI/preprocessing/"
>>>>>>> 84bc2252318cdbb987a6e1b8183d4fa5d540e5cb
    with open(path + data_file, "wb") as f:
        pkl.dump(dictionnary_f, f)
