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
        X = np.matrix('1 2; 3 4; 5 6') # 3 x 2
        theta = np.matrix('2;3') # 2 x 1
        sig = lf.sigmoid(np.dot(X,theta))
        self.assertEqual(np.shape(sig),(3,1))
        self.assertTrue(sig.max() <=1)
        self.assertTrue(sig.min() >=0)

    def test_cost_function(self):
        rand = np.round(np.random.rand(1000,2))
        ones = np.ones((1000,1))
        X = np.append(ones, rand,1) # 1000 x 3

        y = np.ones((1000,1)) # 1000 x 1
        theta = np.matrix('2;3;3') # 3 x 1
        cost = lf.cost_function(X,y,theta)
        self.assertEqual(np.shape(cost),(1,1))
