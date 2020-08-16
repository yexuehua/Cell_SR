import torch
import torch.nn as nn
import math
import pickle


def loss_MSE(x, y, size_average=False):
    e2 = (x-y)**2
    if size_average is True:
        return e2.mean()
    else:
        return e2.sum().div(x.size(0)*2) # ???? why multiply 2 rather than square


def loss_Texture(x, y, nc=3, alpha=1.2, margin=0):
    xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
    yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))

    xi2 = torch.sum(xi*xi, dim=2) # why dim equals 2
    yi2 = torch.sum(yi*yi, dim=2)

    out = nn.functional.relu(yi2.mul(alpha) - xi2 + margin)
    return torch.mean(out)


class Mymodel(nn.Module):
    def __init__(self, scale=1, pretrain="pretrainedmodel.pkl"):
        super(Mymodel, self).__init__()

        self.scale = scale


    def
