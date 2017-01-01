import unittest
import numpy as np
import app.perceptron as perceptron

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

    def test_can_be_trained(self):
        rand = np.random.rand(1000,1)
        y=[]
        map(lambda x: (y.append((x[0] +2) *3)), rand)
        y = np.matrix(y).T
        ones = np.ones((1000,1))
        X = np.append(ones, rand,1)
        self.assertEqual((X[0,1] +2)*3,y[0,0]) # just doublecheck the test itself looks sane
        p = perceptron.Perceptron(2)
        p.train(X,y)


    def test_can_predict(self):
        p = perceptron.Perceptron(3)
        X = np.matrix('-50 -12 1')
        result = p.feed_forward(X)
        print result
