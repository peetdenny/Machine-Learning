import csv
from math import pow,floor
from app import tools

class LinearRegressionMachine:
    trainingset = []
    thetaA=0
    thetaB=32
    alpha = 0.1
    m = 0

    def costFunction(self, thetaA, thetaB):
    #J(thetaA,thetaB) = minimise -> 1/2m sum(i=1->m) h((thetaA,thetaB)(x)-y)^2
        sumOfCosts = 0
        for x,y in self.trainingset:
            h = (thetaA * x) + thetaB * x
            cost = (y-h) # remove square, since we're following the derivative of the cost function
            sumOfCosts += cost
        averageCost = sumOfCosts / m
        return averageCost

    def drange(self,start, stop, step):
        r = start
        while r < stop:
            yield r
            r += step
    def make_cost_function(thetaA):
        return lambda x: costFunction(thetaA, x)

    def gradientDescent(self): # finds the derivate of the cost function with respect to theta_a
        # t0 := t0 - alpha 1/m for i in training set: (h_theta(x_i) - y_i)
        # t1 := t1 - alpha 1/m for i in training set: (h_theta(x_i) - y_i) * x_i
        print 'training model...'
        m = self.m
        alpha = self.alpha
        t0 = 10
        t1 = 10
        def h(x):
            return t0 + (t1 * x)

        temp0 =  alpha * reduce(lambda x,y: x+y, map(lambda x: (h(x[0]) -x[1]), self.trainingset)) / m
        temp1 = alpha * (1/m) * reduce(lambda x,y: x+y, map(lambda x: ((h(x[0]) -x[1]) * x[0]), self.trainingset))
        print 'trained with', temp0, temp1
    def convertCtoF(self, centigrade):
        return floor((centigrade * thetaA) + thetaB)

    def __init__(self):
        tools.read_data('resources/TrainingSet', self.trainingset)
        self.m = len(self.trainingset)
        self.gradientDescent()
        print 'trained model with thetaA of', thetaA
