import numpy as np
from scipy.io import loadmat
import NeuralNetwork

contents = loadmat('dataTest.mat')
X1 = contents['X']
X = (X1 - 127.5)/255
y1 = contents['y']
i = np.arange(3000)
#X = X[(i%300)==0, :]
#y1 = y1[(i%300)==0]
m = X.shape[0]
X = np.concatenate((np.ones((m, 1)), X), axis = 1)

y = np.zeros((m, 10))
for i in range(m):
    y[i,:] = range(10) == y1[i]

contents = loadmat('weights8.mat')
w1 = contents['W1']
w2 = contents['W2']
contents= loadmat('parameters8.mat')
lam = contents['lambda']
print(lam)
net = NeuralNetwork.NN(XCross = X, y1Cross = y1)
(prediction, acc) = net.forwardPass(w1, w2)
print('Accuracy', acc, sep = ' ')
'''
for i in range(m):
    val[i] = pred[i] == y1[i]
#print(val)
(row, col) = np.nonzero(val)
acc = (len(row)/m)*100
print(acc)
'''