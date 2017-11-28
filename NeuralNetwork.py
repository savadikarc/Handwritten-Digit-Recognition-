import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

class NN(object):
    'Main class for NN'
    def __init__(self, X = 0, y = 0, XCross = 0, yCross = 0, y1 = 0, y1Cross = 0):
        self.X = X
        self.y = y
        self.XCross = XCross
        self.yCross = yCross
        self.y1Cross = y1Cross
        self.y1 = y1 
    
    def sigmoid(self, z):
        return(1 / (1 + np.exp(-1*z)))
    
    def lossFunc(self, score2, m, y):
        score2 = score2.clip(min=0.0001)
        return((-1*(np.sum(y * np.log(score2) + (1 - y)*np.log(1 - score2))))/m)
    
    def initWeights(self):
        w1 = np.sqrt(2.0/784)*np.random.normal(loc = 0.0, scale = 1.0, size = (50, 784))
        w1 = np.concatenate((0.01*np.ones((50, 1)), w1), axis = 1)
        w2 = np.sqrt(2.0/50)*np.random.normal(loc = 0.0, scale = 1.0, size = (10, 50))
        w2 = np.concatenate((0.01*np.ones((10, 1)), w2), axis = 1)
        return (w1, w2)
    
    def trainNN(self, w1, w2):
        m = self.X.shape[0]
        mCross = self.XCross.shape[0]
        lam = 12
        alpha = 0.001
        i = 1
        score1 = np.zeros((m, 50))
        score1 = np.concatenate((np.ones((m, 1)), score1), axis = 1)
        score1Cross = np.zeros((mCross, 50))
        score1Cross = np.concatenate((np.ones((mCross, 1)), score1Cross), axis = 1)
        cache1 = np.zeros(w1.shape)
        cache2 = np.zeros(w2.shape)
        decay_rate = 0.9
        eps = 10**(-8)
        loss = 100
        #lossReduce = np.array([24.8, 21.5, 3])
        maxiter = 1000
        decay_alpha = np.array([[89, 1.5], [91, 1.5], [92, 1.1], [93, 1.1], [93.5, 1.1]])
        last_index_alpha_decay = decay_alpha.shape[0] - 1
        ind = 0
        flag = 1
        lossCross = 100
        accCp = 0
        flagacc = 1
        while loss > 0.01:
            
            #Regularization Terms
            reg1 = np.copy(w1)
            reg1[:, 0] = 0
            reg1s = np.sum(np.square(reg1))
            reg2 = np.copy(w2)
            reg2[:, 0] = 0
            reg2s = np.sum(np.square(reg2))
            
            #Forward Pass on Training Set
            a = np.dot(self.X, w1.T)
            score1[:, 1:] = self.sigmoid(a)
            b = np.dot(score1, w2.T)
            score2 = self.sigmoid(b)
            
            #Forward Pass on CV Set
            (score2Cross, accC) = self.forwardPass(w1, w2)
            
            #Store best value for CV set
            if accCp < accC:
                w1best = np.copy(w1)
                w2best = np.copy(w2)
            
            #Loss Function for Training Set
            loss = self.lossFunc(score2, m, self.y) + (lam*(reg1s + reg2s))/(2*m)
            #Loss Function for CV Set
            lossCross = self.lossFunc(score2Cross, mCross, self.yCross)
            
            #RMSProp
            grad2 = (-1*(self.y - score2))
            dw2 = np.dot(grad2.T, score1) + ((lam/(m))*reg2)
            grad1 = np.dot(grad2, w2[:, 1:])
            grad0 = grad1 * score1[:, 1:] *(1 - score1[:, 1:])
            dw1 = np.dot(grad0.T, self.X) + ((lam/(m))*reg1)
            
            #Weight Update
            cache1 = decay_rate*cache1 + (1 - decay_rate) * dw1**2
            cache2 = decay_rate*cache2 + (1 - decay_rate) * dw2**2
            w1 -= alpha*dw1 / (np.sqrt(cache1) + eps)
            w2 -= alpha*dw2 / (np.sqrt(cache2) + eps)
            
            #Print Everything
            print("Epoch:" ,i, "|Loss(Training):", loss, "|Acc(C):", accC, "|Loss(CV):", lossCross, "|alpha:", alpha) 
            
            #lossp = loss
            if accCp < accC:
                accCp = accC
                epoch = i
            
            if (accC >= decay_alpha[(ind, 0)]) and flag == 1:  
                if ind == last_index_alpha_decay:
                    alpha = alpha/decay_alpha[(ind, 1)]
                    flag = 0
                else:
                    alpha = alpha/decay_alpha[(ind, 1)]
                    ind += 1
                
            if i%500 == 0:
                print('Max accuracy achieved ', accCp, 'at epoch ', epoch)
                savemat('weights3.mat', {'W1':w1best, 'W2':w2best})
                savemat('parameters3.mat', {'lambda':lam, 'alpha':alpha, 'maxAcc':accCp})

            if i%maxiter == 0:
                print('Max accuracy achieved ', accCp, 'at epoch ', epoch)
                ch = input('Continue?')
                if ch != 'n':
                    maxiter = int(input('What number of iterations should be run?'))
                elif ch == 'n':
                    break     
            i += 1
            
        return(w1best, w2best, lam, alpha)
    
    def forwardPass(self, w1, w2):
        m = self.XCross.shape[0]
        score1 = np.zeros((m, 50))
        score1 = np.concatenate((np.ones((m, 1)), score1), axis = 1)
        a = np.dot(self.XCross, w1.T)
        score1[:, 1:] = self.sigmoid(a)
        b = np.dot(score1, w2.T)
        score2 = self.sigmoid(b)
        pred = np.argmax(score2, axis = 1)
        val = np.zeros((m, 1))
        for i in range(m):
            val[i] = pred[i] == self.y1Cross[i]
        (row, col) = np.nonzero(val)
        acc = (len(row)/m)*100
        return(score2, acc)
