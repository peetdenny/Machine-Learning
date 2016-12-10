import csv
from math import pow,floor
from app import tools

class LinearRegressionMachine:
    trainingset = []
    thetaA=-1
    thetaB=32

    def costFunction(self, thetaA, thetaB):
    #J(thetaA,thetaB) = minimise -> 1/2m sum(i=1->m) h((thetaA,thetaB)(x)-y)^2
        m = len(self.trainingset)
        sumOfCosts = 0
        for x,y in self.trainingset:
            h = (thetaA * x) + thetaB;
            cost = (y-h) **2
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

    def gradientDescent(self):
        # a super naive implementation. Let's assume that thetaB is known
        print 'training model...'
        thetaB = 32
        h_thetaA = None
        minimum = float('inf')

        aGen = self.drange(0,100,0.1)
        for tA in aGen:
            avCost = self.costFunction(tA,thetaB)
            if avCost < minimum:
#                print avCost, minimum
                minimum = avCost
                h_thetaA = tA
                #print 'new minimum found', h_thetaA, ' at a cost of ', minimum

        #print 'best fit we could find was for ',h_thetaA,' and ',thetaB, 'at a cost of ',minimum
        global thetaA
        global thetaB
        thetaA = h_thetaA

    def convertCtoF(self, centigrade):
        return floor((centigrade * thetaA) + thetaB)

    def __init__(self):
        tools.read_data('resources/TrainingSet', self.trainingset)
        self.gradientDescent()
        print 'trained model with thetaA of', thetaA
