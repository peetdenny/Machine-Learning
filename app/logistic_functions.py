import numpy as np
SIGMOID_BOUNDARY_THRESHOLD=0.5

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def cost_function(X, y, theta):
    m = len(X)
    leftBit = (- y).T * np.log(sigmoid(X.dot(theta)))
    rightBit= (1.0 - y.T) * np.log(1.0 - sigmoid(X.dot(theta)))
    cost = (1.0 / m) * (leftBit - rightBit)
    return cost

def applyBoundary(x):
    if( (x >1.0) | (x <0.0) ):
        raise ValueError("An incorrect value was passed to applyBoundary",x)
    if x > SIGMOID_BOUNDARY_THRESHOLD:
        return 1
    return 0
