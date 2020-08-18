from wavemodel import *
import numpy as np
import cv2
import torch
import os


img = cv2.imread("6_overlay.tif")
wavelet_dec = WaveletTrans()
a = np.expand_dims(img,0)
a = torch.from_numpy(a)
out = wavelet_dec(a)