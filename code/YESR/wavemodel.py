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

class WaveletTransform(nn.Module):
    def __init__(self, scale=1, dec=True, params_path='wavelet_weights_c2.pkl', transpose=True):
        super(WaveletTransform, self).__init__()

        self.scale = scale
        self.dec = dec
        self.transpose = transpose

        ks = int(math.pow(2, self.scale))
        nc = 3 * ks * ks

        if dec:
            self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3,
                                  bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0,
                                           groups=3, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                with open(params_path, 'rb') as f:
                    dct = pickle.load(f, encoding='latin1')
                m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                m.weight.requires_grad = False

    def forward(self, x):
        if self.dec:
            output = self.conv(x)
            if self.transpose:
                osz = output.size()
                # print(osz)
                output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1, 2).contiguous().view(osz)
        else:
            if self.transpose:
                xx = x
                xsz = xx.size()
                xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1, 2).contiguous().view(xsz)
            output = self.conv(xx)
        return output


class VAE(nn.Module):
    def __init__(self,img_size=100*100, hidden1=1024, hidden2=1024, z_dim=24):
        super(VAE, self).__init__()
        self.hidden4 = hidden2
        self.hidden5 = hidden1
        self.output = img_size

        self.encoder = nn.Sequential(
            nn.Linear(img_size,hidden1),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.2)
        )
        self.mean = nn.Linear(hidden2, z_dim)
        self.sigma = nn.Linear(hidden2, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, self.hidden4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden4, self.hidden5),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden5, self.output),
            nn.Sigmoid()
        )


    def forward(self,x):
        en = self.encoder(x)
        coding_mean = self.mean(en)
        coding_sigama = self.sigma(en)
        std = coding_sigama.mul(0.5).exp_()
        noise = torch.autograd.Variable(torch.randn(*coding_mean.size()))
        z = coding_mean + std * noise
        return self.decoder(z), coding_mean, coding_sigama


# EDSR mentioned: no need to the BN for the whole block and relu for the last layer
class _Residual_Block(nn.Module):
    def __init__(self, inc=64, outc=64):
        super(_Residual_Block, self).__init__()

        self.inc = inc
        self.outc = outc

        if inc is not outc:
            self.conv_expand  = nn.Conv2d(in_channels=self.inc, out_channels=self.outc, kernel_size=1, stride=1,
                                          padding=0, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=self.inc, out_channels=self.outc, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=self.inc, out_channels=self.outc, kernel_size=3, stride=1, padding=1,
                               bias=False)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        x = self.relu1(self.conv1(x))
        output = torch.add(self.conv2(x), identity_data)
        return output


def _phase_shift(I, r):
    n, c, a, b = I.size()
    X = torch.reshape(I,(n, r, r, a, b))

    # X -> a x (n, r, r, 1, b)
    X = torch.split(X, a, 3)
    # X -> (n, a*r, r, b)
    X = torch.cat([torch.squeeze(i, dim=3) for i in X], 1)
    # X -> b x (n, a*r, r, 1)
    X = torch.split(X, b, 3)
    # X -> (n, a*r, b*r)
    X = torch.cat([torch.squeeze(i, dim=3) for i in X], 2)
    return torch.reshape(X, (n, 1, a*r, b*r))


def PS(x, r):
    Xc = torch.split(x, 3, 3)
    X = torch.cat([_phase_shift(x, r) for x in Xc], 1)
    return X


class upsample(nn.Module):
    def __init__(self, scale, inc, feature=64):
        super(upsample, self).__init__()

        self.scale = scale
        self.inc = inc
        self.feature = feature

        self.conv1 = nn.Conv2d(in_channels=self.inc, out_channels=self.feature, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        # upsample
        if self.scale == 2:
            ps_feature = self.scale**2
            self.conv2_x2 = nn.Conv2d(in_channels=self.feature, out_channels=ps_feature, kernel_size=3, padding=1,
                                     bias=False)
            self.relu2_x2 = nn.ReLU(inplace=True)

        if self.scale == 3:
            ps_feature = self.scale**2
            self.conv2_x3 = nn.Conv2d(in_channels=self.feature, out_channels=ps_feature, kernel_size=3, padding=1,
                                     bias=False)
            self.relu2_3 = nn.ReLU(inplace=True)

        if self.scale == 4:
            ps_feature = self.scale**2
            self.conv2_x4 = nn.Conv2d(in_channels=self.feature, out_channels=ps_feature, kernel_size=3, padding=1,
                                     bias=False)
            self.relu2_4 = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        if self.scale == 2:
            x = self.relu2_x2(self.conv2_x2(x))
            x = PS(x, 2)

        if self.scale == 3:
            x = self.relu2_x3(self.conv2_x3(x))
            x = PS(x, 3)

        if self.scale == 4:
           x = self.relu2_x4(self.conv2_x4(x))
           x = PS(x, 4)
        return x



class Conv_Net(nn.Module):
    def __init__(self, scale, num_layers):
        super(Conv_Net, self).__init__()

        self.scale = scale
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.res = _Residual_Block(inc=64, outc=64)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        if scale == 2:
            self.up = upsample(2, 64)
        if scale == 3:
            self.up = upsample(3, 64)
        if scale == 4:
            self.up = upsample(4, 64)

    def forward(self, x):
        inImage = x
        x = self.relu1(self.conv1(x))
        for _ in range(self.num_layers):
            x = self.res(x)
        x = torch.add(self.relu2(self.conv2(x)), inImage)
        x = self.up(x)
        return x

