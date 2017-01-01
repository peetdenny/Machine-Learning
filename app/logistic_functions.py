import numpy as np
SIGMOID_BOUNDARY_THRESHOLD=0.5

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def applyBoundary(x):
    if( (x >1.0) | (x <0.0) ):
        raise ValueError("An incorrect value was passed to applyBoundary",x)
    if x > SIGMOID_BOUNDARY_THRESHOLD:
        return 1
    return 0

def perceptron(X, theta):
    return applyBoundary(sigmoid(np.dot(X,theta).sum()))
    
