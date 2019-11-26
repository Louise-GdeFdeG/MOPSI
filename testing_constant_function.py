"""Testing our neural network (nn) trained on a constant function"""

from network import Net, nn, torch
from preprocess_constante import pkl, h
import math
import matplotlib.pyplot as plt
import numpy as np

# Width of the nn
W = 3

# We load the training we done on the net.
net = Net(W)
net.load_state_dict(
    torch.load(
        "/Users/lgainon/Desktop/Cours/Ponts/MOPSI/Network/MOPSI/trained_nn_cst.pt"
    )
)
net.eval()


# Get the data we saved
with open("data_h.pkl", "rb") as f:
    data_h = pkl.load(f)

training_set_h = data_h["train"]
validation_set_h = data_h["valid"]
test_set_h = data_h["test"]


abscissa = np.array([test_set_h[i][0] for i in range(len(test_set_h))])
approximation = [net(torch.FloatTensor([x])).item() for x in abscissa]
h_x = [h(x) for x in abscissa]
print(abscissa[0])
print(h_x[0])
print(approximation[0])
# print(type(abscissa))

plt.clf()  # clean the window
plt.plot(abscissa, h_x, label="Constant function", color="green")
plt.plot(abscissa, approximation, label="Approximation by NN", color="red")
# fig.suptitle('test title', fontsize=20)
# plt.xlabel("xlabel", fontsize=18)
# plt.ylabel("ylabel", fontsize=16)
plt.title("Comparison between the function and its approximation", fontsize=20)
"""plt. axis([0,50,0,10]) """  # absicces de 0 à 50, ordonnées de 0 à 10
plt.grid()  # quadrillage
plt.legend(loc="best")
plt.show()
