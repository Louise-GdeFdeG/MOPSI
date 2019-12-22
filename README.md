# MOPSI - ENPC Project, October 2019

## Aim
The aim of this project is to check which parameters of the model (such as its depth, its width, the learning rate...) has an influence on the quality of the approximation.
We focus on models with linear layers.

## This project contains :
1. A **preprocess folder**: it contains the data and the preprocessing files. There is one `preprocess_function.py` and one `data_function_NbPoints.pkl` per function.  
It is in the preprocess files that the sets of data are created. Here, it includes a constant function, a function that has the same shape of the network, a C-infinite function (polynomial) and a piecewise continuous function. 

2. A `network.py` file : this is where the classes Network are defined. The depth W is taken in argument for the initialization. 
One class is defined for each value of L. For instance, Net3 is the class of NN with L=3, aka 3 layers. 
The forward method describes the behavior of our model. Here, each data passes through each layer, and between each layer the RELU function is applied.

3. A `training.py` file: this is where our model is trained and saved. It is a generic function.

4. A **trained_network folder**: it contains the training files. There is one training file called 'training_function.py' in which we call the code defined in `training.py`. There is only one file for all the functions. To change the function to approximate, one must change the id_function parameter in constant.py . It also contains the trained networks.

5. A `testing.py`file: this is where our model is tested (plots the function, its approximation and the error). It is a generic function.

6.  A **testing_network folder**: it contains the testing file, called 'testing_function.py' . There is one testing file for all the functions in which we call the function defined in `testing.py`. To change the function to approximate, one must change the id_function parameter in constant.py . It also contains the trained networks.

7. A **Meeting_notes folder**: it includes the notes taken during our meetings with our supervisor.

8. A `plotting_e(W).py` file. This is where the average error made by a (W, L) NN is plotted. In this file:
- L is constant and its value is defined in constant.py. 
- W varies and takes its values in the int-list WW, defined in constant.py
- All the other parameters are constant and their value are defined in constant.py

9. A `plotting_e(L).py` file. This is where the average error made by a (W, L) NN is plotted. In this file compared to `plotting_e(W).py`, W and L switch roles:
- W is constant and has the value defined in constant.py. 
- L varies and takes its values in the int-list LL, defined in constant.py
- All the other parameters are constant and their value are defined in constant.py

8. An **Analysis folder**: it contains a report about the tests we've done.

9. A `Notes_PyTorch.docx` file : summary of a quickstarter on PyTorch. 
