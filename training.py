"""Training the neural network (nn) on a constant function"""

from network import Net, nn, torch
from preprocessing.preprocess import pkl, np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt


"""The constants used, the ones we are going to change to assess
their impact on the quality on the model."""
# Width of the nn
W = 3
# Depth of the nn (number of layer)
L = 3
# Number of epoch = number of time our nn trains on the data.
# (it understands fast but we have to explain to it several time)
NB_EPOCH = 10
# The maximum gap we want between the target value and the value given
# by the model
# learning_rate = 0.001
learning_rate = 0.0001


def training(N: int, function: str):
    """ Function used to train our network.
    
    Arguments:
        N {int} -- The number of known points of the function.
        function {str} -- The function identification.
    """
    # Criterion used to measure the error of our model :
    # Mean Square Error
    criterion = nn.MSELoss()

    # network creation
    net = Net(W)

    # The optimizer here is Adam (classic) (eventually to change, after the other tests - gradient descent)
    optim = torch.optim.Adam(net.parameters(), learning_rate)

    # Get the data we saved
    data_file = "data_" + function + "_" + str(N) + ".pkl"
    path = "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI/preprocessing/"
    with open(path + data_file, "rb") as f:
        data = pkl.load(f)

    training_set = data["train"]
    validation_set = data["valid"]
    test_set = data["test"]

    network_file = (
        "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI/trained_network/trained_nn_"
        + function
        + "_"
        + str(N)
        + ".pt"
    )
    loss_train = []
    loss_valid = []
    for epoch in range(NB_EPOCH):
        sum_loss_training = 0
        for i in tqdm(
            range(len(training_set)), desc="Training for epoch " + str(epoch)
        ):
            # Training
            # x and target must be tensors
            x, target = training_set[i][0], training_set[i][1]
            x_tensor, target_tensor = (
                torch.FloatTensor([x]),
                torch.FloatTensor([target]),
            )
            optim.zero_grad()
            output = net(x_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optim.step()
            sum_loss_training += loss.item()
        loss_train.append(sum_loss_training)
        print(sum_loss_training)
        sum_loss_validation = 0
        for i in tqdm(
            range(len(validation_set)), desc="Validation for epoch " + str(epoch)
        ):
            # Validation
            # x and target must be tensors
            x, target = validation_set[i][0], validation_set[i][1]
            x_tensor, target_tensor = (
                torch.FloatTensor([x]),
                torch.FloatTensor([target]),
            )
            optim.zero_grad()
            output = net(x_tensor)
            loss = criterion(output, target_tensor)
            sum_loss_validation += loss.item()
        loss_valid.append(sum_loss_validation)
        print(sum_loss_validation)
        np.random.shuffle(validation_set)
        np.random.shuffle(training_set)
        # we save the training after each epoch
        torch.save(net.state_dict(), network_file)
        # np.random.shuffle(training_set_h) #?
    plt.clf()
    plt.plot(
        [i + 1 for i in range(NB_EPOCH)],
        loss_train,
        "g^",
        label="Sum of loss for training",
    )
    plt.plot(
        [i + 1 for i in range(NB_EPOCH)],
        loss_valid,
        "r--",
        label="Sum of loss for validation",
    )
    plt.xlabel("Number of epochs", fontsize=18)
    plt.ylabel("Loss", fontsize=16)
    plt.title("Loss value depending on the number of epochs done", fontsize=20)
    plt.grid()
    plt.legend(loc="best")
    plt.show()

