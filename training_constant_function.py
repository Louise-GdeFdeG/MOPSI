"""Training the neural network (nn) on a constant function"""

from network import Net, nn, torch
from preprocess import pkl
import math
import matplotlib.pyplot as plt
import numpy as np


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
learning_rate = 0.01


# Criterion used to measure the error of our model :
# Mean Square Error
criterion = nn.MSELoss()

# network creation
net = Net(W)

# The optimizer here is Adam (classic) (eventually to change, after the other tests - gradient descent)
optim = torch.optim.Adam(net.parameters(), learning_rate)

# Get the data we saved
with open("data_h.pkl", "rb") as f:
    data_h = pkl.load(f)

training_set_h = data_h["train"]
validation_set_h = data_h["valid"]
test_set_h = data_h["test"]


# training
for epoch in range(NB_EPOCH):
    for i in range(len(training_set_h)):
        # x and target must be tensors
        x, target = training_set_h[i][0], training_set_h[i][1]
        x_tensor, target_tensor = torch.FloatTensor([x]), torch.FloatTensor([target])
        optim.zero_grad()
        output = net(x_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optim.step()
    # we save the training after each epoch
    torch.save(
        net.state_dict(),
        "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI/trained_nn_cst.pt",
    )
    # np.random.shuffle(training_set_h) #?

