import numpy as np
SIGMOID_BOUNDARY_THRESHOLD=0.5

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def cost_function(X, y, theta):
    m = len(X)
    leftBit = ((np.negative(y)).T * np.log(sigmoid(X.dot(theta))))
    rightBit= (np.ones((len(y),1)) - y).T * np.log(1 - sigmoid(X.dot(theta)))
    print leftBit
    print rightBit
    return (1.0/m) * (leftBit - rightBit)

def applyBoundary(x):
    if( (x >1.0) | (x <0.0) ):
        raise ValueError("An incorrect value was passed to applyBoundary",x)
    if x > SIGMOID_BOUNDARY_THRESHOLD:
        return 1
    return 0

def perceptron(X, theta):
    return applyBoundary(sigmoid(np.dot(X,theta).sum()))
