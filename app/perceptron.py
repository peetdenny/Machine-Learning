import numpy as np
import math
import logistic_functions as lf

class Perceptron:
    theta = []
    alpha = 1e-4 # learning rate

    def __init__(self,size):
        global theta
        global alpha
        theta = np.asmatrix(np.random.rand(size,1))
        alpha = 3000

    def set_theta(self, t):
        global theta
        theta = t

    def feed_forward(self,X):
        return lf.applyBoundary(lf.sigmoid(np.dot(X,theta).sum()))

    def train(self,X,y):
        m = len(X)
        global theta
        global alpha
        cost = float("inf")
        for i in range(10000):
            cost = lf.cost_function(X,y, theta)
            sig = lf.sigmoid(X.dot(theta))
            costs = (1.0/m) * (np.subtract(sig,y))
            rawGrad = np.sum((1.0/m) * np.subtract(sig,y))
            gradient = costs.T.dot(X)
            tempT0 = theta[0,0]
            theta = np.subtract(theta,gradient.T)
            theta[0,0]  = np.subtract(tempT0, (alpha * rawGrad))
        print "min cost found", cost
