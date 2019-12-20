"""Training the neural network (nn) on a constant function"""

from network import Net, nn, torch
from preprocessing.preprocess import pkl, np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt


def training(W: int, L: int, N: int, NB_EPOCH: int, lr: float, function: str):
    """ Function used to train our network.

    Arguments:
        W {int} -- The width of our nn.
        L {int} -- The depth of our nn.
        N {int} -- The number of known points of the function.
        NB_EPOCH {int} -- The number of epoch on which we train our nn.
        lr {float} -- The learning rate we are considering.
        function {str} -- The function identification.
    """
    # Criterion used to measure the error of our model :
    # Mean Square Error
    criterion = nn.MSELoss()

    # network creation
    net = Net(W)

    # The optimizer here is Adam (classic)
    # (eventually to change, after the other tests - gradient descent)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    # Get the data we saved
    data_file = "data_" + function + "_" + str(N) + ".pkl"
    # pour Louise
    path = "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI/"
    # Pour Vivi
    # path = "C:/Users/viniv/OneDrive/Bureau/MOPSI/MOPSI/"
    with open(path + "preprocessing/" + data_file, "rb") as f:
        data = pkl.load(f)

    training_set = data["train"]
    validation_set = data["valid"]
    test_set = data["test"]

    network_file = (
        path
        + "trained_network/trained_nn_"
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

    # Initialize the loss lists, in which we will stock the total loss for
    # each epoch.
    loss_train = []
    loss_valid = []
    abscissa = []
    for epoch in range(NB_EPOCH):
        sum_loss_training = 0
        # Compute the optimization for each point in the training set
        # -------------------------------TRAINING-----------------------------#
        for i in tqdm(
            range(len(training_set)), desc="Training for epoch " + str(1 + epoch)
        ):
            # Training
            # x and target must be tensors
            x, target = training_set[i][0], training_set[i][1]
            x_tensor, target_tensor = (
                torch.FloatTensor([x]),
                torch.FloatTensor([target]),
            )
            # Push the input through the network
            output = net(x_tensor)
            # Reset the gradient buffers
            optim.zero_grad()
            # Compute the loss
            loss = criterion(output, target_tensor)
            # Compute the gradient of the loss and backpropagate it
            loss.backward()
            # Update the network with the chosen optimizer
            optim.step()
            # Add the value of the loss to the partial sum
            sum_loss_training += loss.item()
        # Stock and print the total loss on the period that just ended
        loss_train.append(sum_loss_training)
        print(sum_loss_training)
        sum_loss_validation = 0

        # ----------------------------VALIDATION---------------------------#
        # In this step, we compute the losses on the training set without
        # backprop of the gradient.
        # If the total validation loss has decreased compared to the latest
        # epoch we save the learnings.
        for i in tqdm(
            range(len(validation_set)), desc="Validation for epoch " + str(1 + epoch)
        ):
            # x and target must be tensors
            x, target = validation_set[i][0], validation_set[i][1]
            x_tensor, target_tensor = (
                torch.FloatTensor([x]),
                torch.FloatTensor([target]),
            )
            # Push the input through the network
            output = net(x_tensor)
            # Reset the gradient buffers
            optim.zero_grad()
            # Compute the loss
            loss = criterion(output, target_tensor)
            # Add the value of the loss to the partial sum
            sum_loss_validation += loss.item()
        loss_valid.append(sum_loss_validation)
        print(sum_loss_validation)

        np.random.shuffle(validation_set)
        np.random.shuffle(training_set)
        # We save the training after each epoch only if the validation loss
        # has decreased
        if (epoch > 0 and loss_valid[epoch] < loss_valid[epoch - 1]) or epoch == 0:
            torch.save(net.state_dict(), network_file)
        abscissa.append(epoch)

    plt.clf()
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Number of epochs", fontsize=16)
    ax1.set_ylabel("Sum of loss for training", color="blue", fontsize=16)
    ax1.plot(abscissa, loss_train, "b--")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel(
        "Sum of loss for validation", color="red", fontsize=16
    )  # we already handled the x-label with ax1
    ax2.plot(abscissa, loss_valid, "r--")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title(
        "Approximating "
        + function
        + " function: Sum of loss value for "
        + str(NB_EPOCH)
        + " epochs",
        fontsize=20,
    )
    plt.grid()
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
