# MOPSI - 12/11/2019 Notes

## Elaboration of a strategy for the tests
We will test our network on several functions. It is **essential** to save our data (in case we want to work again on those data later on).  
For each function, we test on the same parameters and change one at the time. In order to have clear methodology, we will work following this instructions :
* W = [5, 10, 50, 100]
* L = [2, 5, 7, 10]
* N = [100, 500, 1000, 2000]
* NB_EPOCH = [1, 5, 10, 50, 100]
* Learning_rate = [1e-4, 1e-3, 1e-2]

Additionally, in order to have a clear organization, we will gather our pre_processing files in one repository. Testing and training must be done in separate files but must be general (we must be able to use them regardless of the function).

Our first objectives are :
1. A constant function
2. A function f defined by f(x) = A_2°ReLU°A_1°ReLU(x) (the same form as the network we are working with)
3. A infinite-continuous function (we'll take a polynomial)
4. A piecewise continuous function.

## Next steps
* Read the documentation about the Adam optimizer, how does it work? Difference with the stochastic gradient ?
* How initialize the coefficients of the layers? For now, they are randomly initialized (this is why sometimes we get really different results for the same function when we run the training process several times). **=> Different initialization should yield to the same result**. We will thus have to verify that for different initial values we found the same minimizer ( = the layers coefficients of the network) at the end.
-> See documentation notes in Notes_PyTorch.docx . 

# MOPSI - 6/11/2019 Notes
## How advanced are we ? 
We have managed to create generic functions coding the creation, training and testing of RELU-linear neural networks. We have first encountered some problems when compiling the code on a constant function. The approximation given by the network was then a constant null function. We now have satisfying results on a constant function. 
We overcame these difficulties by separating the code in different files, and modifying the training process. 

## What is left to do ? 
* Change the display of the approx function and the error. As the values of abscissa are shuffled, displaying the data using lines won't work. There are 2 solutions : display points instead of lines, or reorder the abscissas before displaying. We will probably choose the first solution. 
1. Study the process on polynomial, continue-par-morceaux, and network-like functions. Change W at L constant, and then change L at W constant. Display and save the corresponding courbes.  Discuss the influence of each parameter on the quality of the approximation. 
2. Study the process on polynomial, continue-par-morceaux, and network-like functions. At W and L constant, change lrate, then the size of the training set, then Nepoch. Discuss the influence of each parameter on the quality of the approximation. 

## And then: Image Classification
Once the approximation of a 1D-function is complete, we could tackle image classification. The idea is to build a network that takes an image as an argument and returns a classification value (1 if it thinks the image represents a plane, 2 if it thinks the image represents a cat, etc...) . 
The CIFAR-10 library (https://www.cs.toronto.edu/~kriz/cifar.html) will be useful. A first step would be to dowload the python version of the library and figure out how the images are stored. 
