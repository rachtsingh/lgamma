from torch import autograd
from .internals import polygamma, lgamma

class Beta(autograd.Function):
    def forward(self, a, b):
        beta_ab = (lgamma(a) + lgamma(b) - lgamma(a + b)).exp()
        self.save_for_backward(a, b, beta_ab)
        return beta_ab

    def backward(self, grad_output):
        a, b, beta_ab = self.saved_tensors
        digamma_ab = polygamma(0, a + b)
        return grad_output * beta_ab * (polygamma(0, a) - digamma_ab), grad_output * beta_ab * (polygamma(0, b) - digamma_ab)
