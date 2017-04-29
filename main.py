import numpy as np
#import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import NeuralNetwork

#Import and preprocess Training Data
contents = loadmat('data.mat')
contentsCross = loadmat('dataCross.mat')
X1Cross = contentsCross['X']
XCross = (X1Cross - 127.5)/255
X1 = contents['X']
X = (X1 - 127.5)/255
y1Cross = contentsCross['y']
y1 = contents['y']
mCross = XCross.shape[0]
m = X.shape[0]
XCross = np.concatenate((np.ones((mCross, 1)), XCross), axis = 1)
X = np.concatenate((np.ones((m, 1)), X), axis = 1)

y = np.zeros((m, 10))
for i in range(m):
    y[i,:] = range(10) == y1[i]

yCross = np.zeros((mCross, 10))
for i in range(mCross):
    yCross[i,:] = range(10) == y1Cross[i]

net = NeuralNetwork.NN(X, y, XCross, yCross, y1, y1Cross)
(w1, w2) = net.initWeights()
#contents = loadmat('weights5v1.mat')
#w1 = contents['W1']
#w2 = contents['W2']
(W1, W2, lam, alpha) = net.trainNN(w1, w2)

savemat('weights3.mat', {'W1':W1, 'W2':W2})
savemat('parameters3.mat', {'lambda':lam, 'alpha':alpha})