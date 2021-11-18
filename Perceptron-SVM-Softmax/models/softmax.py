"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = np.random.randn(n_class,3072+1)  # bias 
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        
        g = np.zeros(self.w.shape)
        
        for i in np.arange(X_train.shape[0]):
            
            x = X_train[i]
            y = y_train[i]

            ef = np.exp(np.dot(self.w,x))
            
            g += (np.tile(ef, [3072+1,1]).T * x) / np.sum(ef) #bias
            g[y] -= x 
            
        g /= (i+1)
        
        return g

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        
        X_train -= np.min(X_train) #normalization
        X_train /= np.max(X_train) #normalization
        
        X_train = np.c_[X_train, np.ones((X_train.shape[0],1))] #bias
        
        BatchSize = 200
        Nbatch = X_train.shape[0] // BatchSize
        
        t = 1 #decay
        
        for j in np.arange(self.epochs):
            
            index = np.arange(X_train.shape[0]) #SGD
            np.random.shuffle(index) #SGD
            
            index = np.reshape(index, (Nbatch, index.size//Nbatch)) #mini-batch
            
            for BatchIndex in index:
                
                g = self.calc_gradient(X_train[BatchIndex], y_train[BatchIndex])
                self.w -= self.lr*g
                
            t += 1 #decay
            self.lr *= 1/(100*t) #decay
                
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        
        X_test -= np.min(X_test) #normalization
        X_test /= np.max(X_test) #normalization        
        X_test = np.c_[X_test, np.ones((X_test.shape[0],1))] #bias
        f = np.matmul(self.w,X_test.T)
        y = np.argmax(f,0)
        
        return y
