import torch
import torch.cuda
from _ext import functions

def lgamma(x):
    if x.__class__ == torch.FloatTensor:
        output = torch.FloatTensor(x.size())
        functions.lgamma_py(x, output)
        return output
    elif x.__class__ == torch.DoubleTensor:
        output = torch.DoubleTensor(x.size())
        functions.lgamma_dbl_py(x, output)
        return output
    elif x.__class__ == torch.cuda.FloatTensor:
        output = torch.cuda.FloatTensor(x.size())
        functions.lgamma_cuda(x, output)
        return output
    elif x.__class__ == torch.cuda.DoubleTensor:
        output = torch.cuda.DoubleTensor(x.size())
        functions.lgamma_cuda_dbl(x, output)
        return output
    else:
        raise ValueError

def polygamma(n, x):
    if x.__class__ == torch.FloatTensor:
        output = torch.FloatTensor(x.size())
        functions.polygamma(n, x, output)
        return output
    elif x.__class__ == torch.DoubleTensor:
        output = torch.DoubleTensor(x.size())
        functions.polygamma_dbl(n, x, output)
        return output
    elif x.__class__ == torch.cuda.FloatTensor:
        output = torch.cuda.FloatTensor(x.size())
        functions.polygamma_cuda(n, x, output)
        return output
    elif x.__class__ == torch.cuda.DoubleTensor:
        output = torch.cuda.DoubleTensor(x.size())
        functions.polygamma_cuda_dbl(n, x, output)
        return output
    else:
        raise ValueError
