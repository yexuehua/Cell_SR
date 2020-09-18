from torch.autograd import Variable
from wavemodel import *
import numpy as np
import cv2
import torch
import pandas as pd
from dataset import *
import torch.utils.data
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import torch.nn.functional as F
import os
from tqdm import tqdm

top_path = r"C:\Users\212774000\Documents\python\gitclone\SR\Cell_SR\data"
def test_wavelet():
    img = cv2.imread(r"6_overlay.tif")
    img = np.transpose(img,(2,0,1))
    wavelet_dec = WaveletTransform(scale=1,dec=True)
    a = np.expand_dims(img,0)
    a = torch.from_numpy(a)
    a = Variable(a,requires_grad=False)
    out = wavelet_dec(a.float())
    print(out.size())
    unloader = transforms.ToPILImage()
    for i in range(out.size()[1]):
        output = unloader(out[0][i])
        output.show()

bs = 4

vae = VAE()
criterion_m = nn.MSELoss(size_average=True)
optimizer = torch.optim.Adam(vae.parameters(),lr=1e-3)
data_lists = pd.read_csv("data.csv")
train_lr = list(data_lists["overlay256"])[0:80]
train_hr = list(data_lists["overlay512"])[0:80]
train_set = ImageDatasetFromFile(train_lr,train_hr,top_path)
train_dataloder =torch.utils.data.DataLoader(train_set,batch_size=bs,shuffle=False)
fixed_img = cv2.imread(r"C:\Users\212774000\Documents\python\gitclone\SR\Cell_SR\data\overlay256\8_overlay.tif",cv2.IMREAD_GRAYSCALE)
fixed = cv2.resize(fixed_img,(100,100))
fixed = Image.fromarray(fixed)

def loss_fn(recon_x, x, mu, gamma):
    #BCE = criterion_m(recon_x,x)
    BCE = F.binary_cross_entropy(recon_x,x,size_average=False)
    KLD = -0.5 * torch.sum(1 + gamma - mu**2 -gamma.exp())
    return BCE.mul(0.3)+KLD.mul(0.9)


def test_VAE():
    epochs = 1000
    for epoch in range(epochs):
        for idx,(lr,hr) in enumerate(train_dataloder):
            imgs = Variable(lr.view(lr.size(0),-1))
            target = Variable(hr.view(hr.size(0),-1))
            recons_img,mu,gamma = vae(imgs)
            loss = loss_fn(recons_img, target, mu, gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx%50 == 0:
                print("Epoch[{}/{}] Loss:{:.3f}".format(epoch+1,epochs,loss.data/bs))
                #save_image(target.view(-1, 1, 100, 100), f'recon/target_img_{epoch}_{idx}.png')
                #save_image(imgs.view(-1,1,100,100),f'recon/ori_img_{epoch}_{idx}.png')
                save_image(recons_img.view(-1,1,100,100),f'recon/recon_img_{epoch}_{idx}.png')
test_VAE()
