# Handwritten-Digit Recognition

I used the MNIST handwritten digit dataset.

The image is of size 28x28, which is stored as a row vector in a .mat file. A total of 10000 iamges (1000 of each digit) are divided into Training Set (5000), Cross Validation Set (3000), and Test Set (2000).

Specifications of FNN:

 1. Number of input units - 784 + bias = 785.
 2. Number of hidden layer(s) - 1 having 50 neurons.
 3. Number of output classes - 10 each class corresponding to a digit.
 
main.py contains the script to be run.

NeuralNetwork contains the class NN for the FNN.

predictDigit.py contains the code to check the prediction accuracy on any datadet specified in the 'loadmat' function.

weights6.mat contains the trained weights and parameters6.py contains he value of regularization parameter lambda used, as well as the last decayed learning rate alpha.

data.mat, dataCross.mat and dataTest.mat are the training, cross validation and test datasets respectively.
