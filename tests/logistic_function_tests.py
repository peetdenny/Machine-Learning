import unittest
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
