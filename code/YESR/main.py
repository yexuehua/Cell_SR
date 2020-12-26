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

"""
data preparation for training and test
"""
top_path = r"C:\Users\212774000\Documents\python\gitclone\SR\Cell_SR\data"
bs = 4
conv_model = Conv_Net(2, 16)
criterion_m = nn.MSELoss(size_average=True)
optimizer = torch.optim.Adam(conv_model.parameters(),lr=1e-3)
data_lists = pd.read_csv("data.csv")
train_lr = list(data_lists["overlay256"])[0:80]
train_hr = list(data_lists["overlay512"])[0:80]
train_set = ImageDatasetFromFile(train_lr,train_hr,top_path)
train_dataloder =torch.utils.data.DataLoader(train_set,batch_size=bs,shuffle=False)

"""
Define the loss function
"""
def loss_fn(recon_x, x):
    #BCE = criterion_m(recon_x,x)
    BCE = F.binary_cross_entropy(recon_x,x,size_average=False)
    return BCE.mul(0.3)


"""
Begin to train model
"""
def train():
    epochs = 10000
    for ep in range(epochs):
        for idx,(lr,hr) in enumerate(train_dataloder):
            imgs = Variable(lr.view(lr.size(0),-1))
            target = Variable(hr.view(hr.size(0),-1))
            recons_img = conv_model(imgs)
            loss = loss_fn(recons_img, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx%50 == 0:
                print("Epoch[{}/{}] Loss:{:.3f}".format(ep+1,epochs,loss.data/bs))
                #save_image(target.view(-1, 1, 100, 100), f'recon/target_img_{epoch}_{idx}.png')
                #save_image(imgs.view(-1,1,100,100),f'recon/ori_img_{epoch}_{idx}.png')
                save_image(recons_img.view(-1,1,100,100),f'recon/recon_img_{ep}_{idx}.png')




