import unittest
from app.linear_regression import LinearRegressionMachine
from app import tools

class LinearRegressionTests(unittest.TestCase):
    def test_trained_regression(self):
        test_data =[]
        tools.read_data('resources/TestSet',test_data)
        print 'loaded', len(test_data), 'tests'
        m = LinearRegressionMachine();
        assertions=0
        for pair in test_data:
            self.assertEqual(m.convertCtoF(pair[0]), pair[1]);
            assertions +=1
        print assertions, 'assertions'
