"""Testing our neural network (nn) trained on a constant function"""

from network import Net2, Net3, Net4, Net5, Net7, Net10, nn, F, torch
from preprocessing.preprocess import pkl
from preprocessing.preprocess_constant import h
import math
import matplotlib.pyplot as plt
import numpy as np


def testing(W: int, L: int, N: int, NB_EPOCH: int, lr: float, function: str):
    """ Display the function and its approximation by our network.
    

    Arguments:
        W {int} -- The width of our nn.
        L {int} -- The depth of our nn.
        N {int} -- The number of known points of the function.
        NB_EPOCH {int} -- The number of epoch on which we train our nn.
        lr {float} -- The learning rate we are considering.
        function {str} -- The function identification.
    Returns : 
        totalerror {tensor} The total error calculated on the testing set
    """

    # -----------------------IMPORT TRAINED NETWORK---------------------------#
    # Pour Louise :
    # path = "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI/"
    # Pour Vivi :
    path = "C:/Users/viniv/OneDrive/Bureau/MOPSI2/"
    # Pour Jean :
    # path = "/Users/Jean/Documents/Ponts/MOPSI/MOPSI/"
    network_file = (
        "trained_nn_"
        + function
        + "_"
        + str(W)
        + "_"
        + str(L)
        + "_"
        + str(N)
        + "_"
        + str(NB_EPOCH)
        + "_"
        + str(lr)
        + ".pt"
    )

    # We load the training we done on the net.
    # Network creation : call the class that matches the L argument
    if L == 2:
        net = Net2(W)
    elif L == 3:
        net = Net3(W)
    elif L == 4:
        net = Net4(W)
    elif L == 5:
        net = Net5(W)
    elif L == 7:
        net = Net7(W)
    elif L == 10:
        net = Net10(W)
    else:
        print("Error: L given in argument matches no Net class")
    net.load_state_dict(torch.load(path + "trained_network/" + network_file))
    # Set the net on evaluation mode
    net.eval()

    # Get the data we saved
    data_file = "data_" + function + "_" + str(N) + ".pkl"
    with open(path + "preprocessing/" + data_file, "rb") as f:
        data = pkl.load(f)

    training_set = data["train"]
    validation_set = data["valid"]
    test_set = data["test"]

    # ---------------------------TESTING ----------------------------------- #

    criterion = nn.MSELoss()

    # Initialize the lists for plotting
    error = []
    abscissa = []
    # Values of the target function
    h_x = []
    # Values of the approximation given by the network
    approximation = []

    # COmpute the approximation and the error fot each value in test_set
    for i in range(len(test_set)):
        x, target = test_set[i][0], test_set[i][1]
        abscissa.append(x)
        h_x.append(target)
        output = net(torch.FloatTensor([x]))
        approximation.append(output.item())
        error.append(criterion(output, torch.FloatTensor([target])))

    plt.clf()
    plt.figure()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Abscissa", fontsize=14)
    ax1.set_ylabel("Values", fontsize=14)
    ax1.plot(abscissa, h_x, "b^", label="Objective function")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.plot(abscissa, approximation, "r+", label="Approximation by NN")

    plt.title(
        "Comparison between "
        + function
        + " function and its approximation. W="
        + str(W)
        + "L="
        + str(L),
        fontsize=16,
    )
    plt.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(loc='upper right')
    plt.show()

    # plt.plot(abscissa, error, "g+")
    # plt.title(
    #     "Error between " + function + " function and the approximation", fontsize=16
    # )
    plt.plot(abscissa, error, "go")
    plt.title(
        "Error between "
        + function
        + " function and the approximationW="
        + str(W)
        + "L="
        + str(L),
        fontsize=16,
    )

    plt.show()
    totalerror = sum(error).item()
    print("Total error: ", totalerror)
    return totalerror
