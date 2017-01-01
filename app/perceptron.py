import numpy as np
import math
import logistic_functions as lf

class Perceptron:
    theta = []

    def __init__(self,size):
        global theta
        theta = np.random.rand(size,1)
    def set_theta(self, t):
        global theta
        theta = t

    def feed_forward(self,X):
        return lf.applyBoundary(lf.sigmoid(np.dot(X,theta).sum()))



    def train(self,X,y):
        # cost function: perform squared error on h_theta(X) vs y
        cost = lf.cost_function(X,y, theta)
        #TODO implement gradient and gradient descent
        print "COST",cost
