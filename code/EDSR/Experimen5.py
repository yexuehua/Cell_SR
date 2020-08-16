from model import EDSR
import pydicom
import numpy as np
import argparse
import os
from scipy import misc
import time


#super resolution using NN model
'''
parser = argparse.ArgumentParser()
parser.add_argument("--imsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--image")

args = parser.parse_args()
down_size = args.imsize//args.scale
network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
network.resume(args.savedir)

def normal(pixels):
    high,wid = pixels.shape
    mins = np.min(pixels)
    maxs = np.max(pixels)
    for i in range(high):
        for j in range(wid):
            pixels[i][j] = int((pixels[i][j]-mins)*(255/(maxs-mins)))
    return pixels

top_path = r"D:\python\EDSR-Tensorflow-master\result\Experiment5"
outimg_dir = os.path.join(top_path,"NN_x2out")
inimg_dir = os.path.join(top_path,"origin_img")
imgs = os.listdir(top_path)
if not os.path.exists(outimg_dir):
    os.mkdir(outimg_dir)
if not os.path.exists(inimg_dir):
    os.mkdir(inimg_dir)
time_start = time.time()
for x in imgs:
    imgpath = os.path.join(top_path,x)
    img = pydicom.read_file(imgpath)
    inputs = img.pixel_array
    inputs = normal(inputs)
    x = x.replace('dcm','jpg')
    in_name = 'origin'+x
    in_imag_path = os.path.join(inimg_dir,in_name)
    #misc.imsave(in_imag_path,inputs)
    inputs = np.expand_dims(inputs,axis=2)
    outputs = network.predict(inputs)
    outputs = outputs[:,:,0]
    out_name = 'nn_x2out'+x
    out_img_path = os.path.join(outimg_dir,out_name)
    misc.imsave(out_img_path,outputs)
time_end = time.time()
print("the whole time is ",time_end-time_start,"s")
'''

#super resolution using interpolation algorithm
'''
top_path = r"D:\python\EDSR-Tensorflow-master\result\Experiment5"
imgs_path = top_path+'\origin_img'
imgs = os.listdir(imgs_path)
out_path = os.path.join(top_path,'inter_x2out')
if not os.path.exists(out_path):
    os.mkdir(out_path)
time_start = time.time()
for img in imgs:
    img_path = os.path.join(imgs_path,img)
    img_data = misc.imread(img_path)
    x,y = img_data.shape
    img_data = misc.imresize(img_data,(x*2,y*2),interp='bicubic')
    out_name = img.replace("origin","bicubic")
    out_img_path = os.path.join(out_path,out_name)
    misc.imsave(out_img_path,img_data)
time_end = time.time()
print("the whole time is ",time_end-time_start,"s")

'''

# calculate the SNR value for every image

def calculate(imgs,path):
    SNR={}
    for img in imgs:
        img_path = os.path.join(path,img)
        img_data = misc.imread(img_path)
        signal_img = img_data[215*2:235*2,95*2:120*2]
        sig_sum = np.sum(signal_img**2)
        noise_img = img_data[200*2:220*2,10*2:35*2]
        noi_sum = np.sum(noise_img**2)
        inti_SNR = -20*np.log10(sig_sum/noi_sum)
        SNR[img]=inti_SNR
        # SNR.append(img)
        # SNR.append(inti_SNR)
    return SNR


top_path = r"D:\python\EDSR-Tensorflow-master\result\Experiment5"
inter_imgs_path = os.path.join(top_path,'inter_x2out')
NN_imgs_path = os.path.join(top_path,'NN_x2out')
inter_imgs = os.listdir(inter_imgs_path)
NN_imgs = os.listdir(NN_imgs_path)
Interp_SNR = calculate(inter_imgs,inter_imgs_path)
NN_SNR = calculate(NN_imgs,NN_imgs_path)
Interp_value = list(Interp_SNR.values())
Interp_mean = np.mean(Interp_value)
NN_value = list(NN_SNR.values())
NN_mean = np.mean(NN_value)
print(NN_SNR)
print(Interp_SNR)
print(Interp_mean,'\n',NN_mean)

# evaluate by mean and variance
'''
def calculate(imgs,path):
    SNR=[]
    for img in imgs:
        img_path = os.path.join(path,img)
        img_data = misc.imread(img_path)
        signal_img = img_data[163*2:198*2,230*2:255*2]
        sig_sum = np.sum(signal_img**2)
        noise_img = img_data[163*2:198*2,205*2:230*2]
        noi_sum = np.sum(noise_img**2)
        inti_SNR = -20*np.log10(sig_sum/noi_sum)
        SNR.append(img)
        SNR.append(inti_SNR)
    return SNR

top_path = r"D:\python\EDSR-Tensorflow-master\result\Experiment5"
inter_imgs_path = os.path.join(top_path,'inter_x2out')
NN_imgs_path = os.path.join(top_path,'NN_x2out')
inter_imgs = os.listdir(inter_imgs_path)
NN_imgs = os.listdir(NN_imgs_path)
Interp_SNR = []
NN_SNR = []
'''

