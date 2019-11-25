"""Testing what parameters influence the approximation of a function
by a neural network (nn)"""

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
# The maximum gap we want between the target value and the value given
# by the model
# learning_rate = 0.001
learning_rate = 0.01


# Criterion used to measure the error of our model :
# Mean Square Error
criterion = nn.MSELoss()

# The optimizer here is Adam (classic) (eventually to change, after the other tests - gradient descent)
optim = torch.optim.Adam(net.parameters(), learning_rate)


# Get the data we saved
with open("data_f.pkl", "rb") as f:
    data_f = pkl.load(f)

training_set_f = data_f["train"]
validation_set_f = data_f["valid"]
test_set_f = data_f["test"]
error_list = []
xW = []
for W in range(3, 11):
    print(W)
    # network creation
    net = Net(W)
    # The optimizer here is Adam (classic)
    optim = torch.optim.Adam(net.parameters(), learning_rate)
    # training
    for i in range(len(training_set_f)):
        # x and target must be tensors
        x, target = training_set_f[i][0], training_set_f[i][1]
        x_tensor, target_tensor = torch.FloatTensor([x]), torch.FloatTensor([target])
        optim.zero_grad()
        output = net(x_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optim.step()

    # Testing
    error_W = 0
    N = len(test_set_f)
    for i in range(N):
        x, target = test_set_f[i][0], test_set_f[i][1]
        x_tensor, target_tensor = torch.FloatTensor([x]), torch.FloatTensor([target])
        # preparing the calculus of the error function
        floatoutput = float(net(x_tensor))
        error_W += (floatoutput - target) ** 2
    error_W = math.sqrt((1 / N) * error_W)
    # Adding value to the error vector
    error_list.append(error_W)
    xW.append(W)

# Plotting the error as a function of W
print(error_list)
errW = np.array(error_list)
abscisse = np.array(xW)
plt.plot(abscisse, errW)
