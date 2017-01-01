import unittest
import numpy as np
import app.logistic_functions as lf

class LogisticFunctionTests(unittest.TestCase):
    def assertShesASigmoidGoodun(self, x ,exp):
        y = lf.sigmoid(x)
        if(exp):
            self.assertTrue(y>0.9999)
        else:
            self.assertTrue(y < 0.0001)

    def test_sigmoid_function(self):
        self.assertShesASigmoidGoodun(-55, 0)
        self.assertShesASigmoidGoodun(55, 1)
        self.assertShesASigmoidGoodun(-150, 0)
        self.assertShesASigmoidGoodun(150, 1)
        self.assertShesASigmoidGoodun(-100000, 0)
        self.assertShesASigmoidGoodun(100000, 1)
        self.assertEqual(lf.sigmoid(0),0.5)

    def test_works_with_matrices(self):
        X = np.matrix('1 2; 3 4; 5 6')
        theta = np.matrix('2 2; 3 3')
        sig = lf.sigmoid(np.dot(X,theta))
        self.assertTrue(sig.max() <=1)
        self.assertTrue(sig.min() >=0)

class PerceptronTests(unittest.TestCase):
    def test_perceptron_firing(self):
        self.assertEqual(lf.perceptron([1], [1]),1)
        self.assertEqual(lf.perceptron([0],[0]),0)
        X = np.matrix('1 2 3 4;5 6 7 8')
        theta = np.matrix('2;2;2;2')
        y = lf.perceptron(X,theta)
        self.assertEqual(lf.perceptron(X,theta),1)

        X = np.matrix('1 2 3 4;5 6 7 8')
        theta = np.matrix('-2;-2;-2;-2')
        y = lf.perceptron(X,theta)
        self.assertEqual(lf.perceptron(X,theta),0)
