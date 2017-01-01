import unittest
import numpy as np
import app.perceptron as perceptron
import app.logistic_functions as lf

class PerceptionTests(unittest.TestCase):
    def test_logistic_correctness(self):
        p = perceptron.Perceptron(2)
        p.feed_forward(np.random.rand(3,2)) # also tests that theta is initialised to right size
        p.set_theta([1])
        self.assertEqual(p.feed_forward([1]),1)
        p.set_theta([0])
        self.assertEqual(p.feed_forward([0]),0)

        X = np.matrix('1 2 3 4;5 6 7 8')
        theta = np.matrix('2;2;2;2')
        p.set_theta(theta)
        y = p.feed_forward(X)
        self.assertEqual(p.feed_forward(X),1)

        X = np.matrix('1 2 3 4;5 6 7 8')
        theta = np.matrix('-2;-2;-2;-2')
        p.set_theta(theta)
        y = p.feed_forward(X)
        self.assertEqual(p.feed_forward(X),0)

    def populateY(self, X):
        y = np.ones((1000,1))
        for i in range(1000):
            t = X[i,1] + X[i,2]
            y[i,0] =  (1 if t==2 else 0)
        return y

        for i in range(1000):
            self.assertEqual((X[i,1]==1) & (X[i,2]==1), y[i,0])

    def boolean_looks_good(self, p, inputs, expected):
        inputM = np.matrix(inputs)
        output = p.feed_forward(inputM)
        self.assertEquals(expected, output)

    def test_can_be_trained(self):
        rand = np.round(np.random.rand(1000,2))
        ones = np.ones((1000,1))
        X = np.append(ones, rand,1)
        y = np.ones((1000,1))
        y = self.populateY(X)
        p = perceptron.Perceptron(3)
        cost = p.train(X,y)
        self.assertTrue(cost < 0.0001)

        self.boolean_looks_good(p, '1 1 1', 1)
        self.boolean_looks_good(p, '1 0 1', 0)
        self.boolean_looks_good(p, '1 1 0', 0)
        self.boolean_looks_good(p, '1 0 0 ', 0)
