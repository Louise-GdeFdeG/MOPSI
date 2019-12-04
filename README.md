# MOPSI - ENPC Project, October 2019

## Aim
The aim of this project is to check which parameters of the model (such as its depth, its width, the learning rate...) has an influence on the quality of the approximation.
We focus on models with linear layers.

## This project contains :
1. A **preprocess folder**: it contains the data and the preprocessing files. There is one `preprocess_function.py` and one `data_function_NbPoints.pkl` per function.  
It is in the preprocess files that the sets of data are created. Here, it includes a constant function, a function that has the same shape of the network, a C-infinite function (polynomial) and a piecewise continuous function. 

2. A `network.py` file : this is where the class Network is defined. The depth W is taken in argument for the initialization. To modify L (the number of layer) it is necessary to do it "by hand", by modifying the init method and the forward method (add a line). The forward method describes the behavior of our model. Here, each data passes through each layer, and between each layer the RELU function is applied.

3. A `training.py` file: this is here our model is trained and saved. It is a generic function.

4. A **trained_network folder**: it contains the training files. There is one training file per function in which we call the function defined in `training.py`. It also contains the trained networks.

5. A `testing.py`file: this is here our model is tested (plots the function, its approximation and the error). It is a generic function.

6.  A **testing_network folder**: it contains the testing files. There is one testing file per function in which we call the function defined in `testing.py`.

7. A **Meeting_notes folder**: it includes the notes taken during our meetings with our supervisor.

8. An **Analysis folder**: it contains a report about the tests we've done.