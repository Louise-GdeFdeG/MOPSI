"""Testing our neural network (nn) trained on a constant function"""

from network import Net, nn, torch
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
    """
    path = "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI/"
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
    net = Net(W)
    net.load_state_dict(torch.load(path + "trained_network/" + network_file))
    net.eval()

    # Get the data we saved
    data_file = "data_" + function + "_" + str(N) + ".pkl"
    with open(path + "preprocessing/" + data_file, "rb") as f:
        data = pkl.load(f)

    training_set = data["train"]
    validation_set = data["valid"]
    test_set = data["test"]

    # Testing
    error = []
    criterion = nn.MSELoss()
    h_x = []
    abscissa = []
    approximation = []
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
    ax1.set_ylabel(function + " function", color="blue", fontsize=14)
    ax1.plot(abscissa, h_x, "b^")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel(
        "Approximation by NN", color="red", fontsize=14
    )  # we already handled the x-label with ax1
    ax2.plot(abscissa, approximation, "r--")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Comparison between the function and its approximation", fontsize=16)
    plt.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()

    plt.plot(abscissa, error, color="green")
    plt.title("Error between the function and the approximation", fontsize=16)
    plt.show()
