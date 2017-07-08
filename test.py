import torch
import torch.cuda
from torch.autograd import Variable, gradcheck

import unittest

from functions.beta import Beta
from functions.digamma import Digamma

tensorType = torch.cuda.DoubleTensor

class TestDigammaGrads(unittest.TestCase):
    def test_many_times(self):
        input = (Variable(tensorType(50, 100).uniform_(), requires_grad=True),)
        self.assertTrue(gradcheck(Digamma(), input, eps=1e-6, atol=1e-3))

class TestBetaGrads(unittest.TestCase):
    def test_many_times(self):
        a = Variable(tensorType(71, 23).uniform_() * 10, requires_grad=True)
        b = Variable(tensorType(71, 23).uniform_() * 10, requires_grad=True)
        result = gradcheck(Beta(), (a, b), eps=1e-6, atol=1e-3)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
