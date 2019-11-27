# MOPSI - ENPC Project, October 2019

## Aim
The aim of this project is to check which parameters of the model (such as its depth, its width, the learning rate...) has an influence on the quality of the approximation.
We focus on models with linear layers.

## This project contains :
1. A **preprocess repository**: it contains the data and the preprocessing files. There is one `preprocess_function.py` and one `data_function.pkl` per function.  
It is in the preprocess files that the sets of data are created. Here, it includes a constant function, a function that has the same shape of the network, a C-infinite function (polynomial) and a piecewise continuous function. 

2. A `network.py` file : this is where the class Network is defined. The depth W is taken in argument for the initialization. The number of linear layer L is here equal to 3. To modify L it is necessary to do it "by hand", by modifying the init method and the forward method (add a line). The forward method describes the behavior of our model. Here, each data passes through each layer, and between each layer the RELU function is applied.

3. A `training.py` file: this is here our model is trained and we where we can study its effectiveness.

4. `data_f.pkl` and `data_g.pkl` : our data sets.