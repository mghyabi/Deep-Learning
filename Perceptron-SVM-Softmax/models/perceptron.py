"""Perceptron model."""

import numpy as np


from operator import add, sub
class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = np.random.randn(n_class,3072+1)  # bias 
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        
        
        t = 1 #decay

        X_train -= np.min(X_train) #normalization
        X_train /= np.max(X_train) #normalization
        
        X_train = np.c_[X_train, np.ones((X_train.shape[0],1))] #bias
        
        for j in np.arange(self.epochs):
            
            index = np.arange(X_train.shape[0]) # SGD
            np.random.shuffle(index) #SGD
            
            for i in index:

                x = X_train[i]
                y = y_train[i]

                f = np.dot(self.w,x)

                if np.argmax(f) != y:

                    self.w[y] += self.lr * np.sum(f > f[y]) * x
                    self.w -= self.lr * np.tile(1*(f > f[y]),[3072+1,1]).T * x #bias
                
            t += 1 #decay
            self.lr *= 1/(200*t) #decay

        pass

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
