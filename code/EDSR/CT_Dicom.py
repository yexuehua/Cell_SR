from model import EDSR
import scipy.misc
import numpy as np
import argparse
import data
import os
import pydicom
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/General-100")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--outdir",default="out")
parser.add_argument("--image")
args = parser.parse_args()
if not os.path.exists(args.outdir):
	os.mkdir(args.outdir)
down_size = args.imgsize//args.scale
network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
network.resume(args.savedir)


def normal(pixels):
	high,wid = pixels.shape
	for i in range(high):
		for j in range(wid):
			pixels[i][j] = int(pixels[i][j]*(255/1023))
#			pixels[i][j] = int((pixels[i][j]+2048)*(256/4096))# for ct
	return pixels

def denormal(pixels):
	high,wid = pixels.shape
	for i in range(high):
		for j in range(wid):
			pixels[i][j] = int(pixels[i][j]*(1023/255))
#			pixels[i][j] = int((pixels[i][j]-127)*(2048/127))
	return pixels

top_path = r"D:\python\EDSR-Tensorflow-master\Experiment2"
outimg_dir = os.path.join(top_path,"out_dcm")
imgs = os.listdir(top_path)
if not os.path.exists(outimg_dir):
	os.mkdir(outimg_dir)
for x in imgs:
	imgpath = os.path.join(top_path,x)
	img = pydicom.read_file(imgpath)
	inputs = img.pixel_array
	inputs = normal(inputs)
	inputs = np.expand_dims(inputs, axis=2)
	inputs = np.concatenate((inputs, inputs, inputs), axis=-1)
	outputs = network.predict(inputs)
	out_name = "out_" + x
	out_img_path = os.path.join(outimg_dir,out_name)
	outputs=outputs[:,:,0]
	outputs=denormal(outputs)
	outputs_img = np.uint16(outputs)
	img.PixelData = outputs_img.tobytes()
	img.Rows,img.Columns = outputs.shape
	pixel_spacing = img.PixelSpacing
	ps1 = float(pixel_spacing[0])/2
	ps2 = float(pixel_spacing[1])/2
	pixel_spacing = [str(ps1),str(ps2)]
	img.PixelSpacing = pixel_spacing
	img.save_as(out_img_path)




# top_path = r"D:\yexuehua\data\Experiment"
# mods = os.listdir(top_path)
# for mod in mods:
# 	datapath = os.path.join(top_path,mod)
# 	outimg_dir = os.path.join(datapath,"2xout_dcm")
# 	imgs = os.listdir(datapath)
# 	if not os.path.exists(outimg_dir):
# 		os.mkdir(outimg_dir)
# 	for x in imgs:
# 		imgpath = os.path.join(datapath,x)
# 		img = pydicom.read_file(imgpath)
# 		inputs = img.pixel_array
# 		inputs = normal(inputs)
# 		inputs = np.expand_dims(inputs, axis=2)
# 		inputs = np.concatenate((inputs, inputs, inputs), axis=-1)
# 		outputs = network.predict(inputs)
# 		out_name = "out_" + x
# 		out_img_path = os.path.join(outimg_dir,out_name)
# 		outputs=outputs[:,:,0]
# 		outputs=denormal(outputs)
# 		outputs_img = np.uint16(outputs)
# 		img.PixelData = outputs_img.tobytes()
# 		img.Rows,img.Columns = outputs.shape
# 		pixel_spacing = img.PixelSpacing
# 		ps1 = float(pixel_spacing[0])/2
# 		ps2 = float(pixel_spacing[1])/2
# 		pixel_spacing = [str(ps1),str(ps2)]
# 		img.PixelSpacing = pixel_spacing
# 		img.save_as(out_img_path)


