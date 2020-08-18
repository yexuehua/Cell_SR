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


class WaveletTrans(nn.Module):
    def __int__(self, scale=1, dec=True, para_path='wavelet_weights_c2.pkl', transpose=True):
        super(WaveletTrans, self).__init__()

        self.scale = scale
        self.dec = dec
        self.para_path = para_path
        self.transpose = transpose

        ks = int(math.pow(2, scale))
        nc = 3 * ks * ks

        if self.ec:
            self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0,groups=3, bias=None)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0, groups=3, bias=None)

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                with open(self.para_path,"rb") as f:
                    dct =pickle.load(f,encoding="latin1")
                m.weight.data = torch.from_numpy(dct["rec%"%ks])
                m.weight.requires_grad(False)

    def forward(self, x):
        if self.dec:
            output = self.conv(x)
            if self.transpose:
                osz =output.size()
                output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1,2).contiguous().view(osz)
        else:
            if self.transpose:
                xx = x
                xsz = xx.size()
                xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1,2).contiguous().view(xsz)
            output = self.conv(xx)

        return output





class Mymodel(nn.Module):
    def __init__(self, scale=1, pretrain="pretrainedmodel.pkl"):
        super(Mymodel, self).__init__()

        self.scale = scale

