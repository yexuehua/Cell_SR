# python
import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import log10
# pytorch
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
# DIY
from wavemodel import *
from dataset import *

parse = argparse.ArgumentParser()
parse.add_argument("--cuda", action="store_true", help="enable cuda")
parse.add_argument("--top_path", default=r"C:\Users\212774000\Documents\python\gitclone\SR\Cell_SR\data", type=str,
                   help="path to the whole data")
parse.add_argument("--bs", default=4, type=int, help="train batch size")
parse.add_argument("--tbs", default=1, type=int, help="test batch size")
parse.add_argument("--Epoch", default=5000, type=int, help="ep")
parse.add_argument("--test_iter", default=50, type=int, help="test iterator steps")


def save_image(img, path):
    img_save = img.to("cpu")
    img_data = img_save.data.numpy().astype(np.float32)
    img_data = np.squeeze(img_data)
    cv2.imwrite(path, img_data)

def main():
    # define some config
    opt = parse.parse_args()
    top_path = opt.top_path
    bs = opt.bs
    if torch.cuda.is_available() and not opt.cuda:
        print("Warning: Detected a cuda device, you should use --cuda")

    # create model object
    conv_model = Conv_Net(2, 16)
    mae = nn.L1Loss()
    mse = nn.MSELoss()

    # add to cuda device
    device = torch.device("cuda")
    cudnn.benchmark = True
    if opt.cuda:
        conv_model.to(device)
        mae.to(device)

    optimizer = torch.optim.Adam(conv_model.parameters(), lr=1e-3)

    # create the dataloader
    data_lists = pd.read_csv("data.csv")
    # training data
    train_lr = list(data_lists["overlay256"])[0:80]
    train_hr = list(data_lists["overlay512"])[0:80]
    train_set = ImageDatasetFromFile(train_lr, train_hr, top_path)
    train_dataloder = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
    # test data
    test_lr = list(data_lists["overlay256"])[80:]
    test_hr = list(data_lists["overlay512"])[80:]
    test_set = ImageDatasetFromFile(test_lr, test_hr, top_path)
    test_dataloder = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False)

    # note that if there are no batch normal and dropout layer in your model,it's no need to do model.train()/eval()
    # conv_model.train()
    print("---------start training--------")
    # ------------------train by steps-------------------
    for ep in range(opt.Epoch):
        if ep % opt.test_iter == 0:
            sum_psnr = 0
            for test_batch in test_dataloder:
                img, target = test_batch[0], test_batch[1]
                if opt.cuda:
                    img.to(device)
                    target.to(device)
                pred = conv_model(img)
                psnr = 10 * log10(1/mse(pred, target).item())
                sum_psnr += psnr
            save_image(pred, "recon/test/{}-ep_pred.png".format(ep))
            save_image(img, "recon/test/{}-ep_img.png".format(ep))
            save_image(target, "recon/test/{}-ep_target.png".format(ep))
            print("===> Avg.PSNR:{:.4f} dB".format(sum_psnr/len(test_dataloder)))
        for idx, batch in enumerate(train_dataloder):
            img, target = batch[0], batch[1]
            recons_img = conv_model(img)
            loss = mae(recons_img, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("===>loss: {:.4f}".format(loss.item()))