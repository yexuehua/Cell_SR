from model import EDSR
import numpy as np
import argparse
import os
import pydicom
from scipy import misc

parser = argparse.ArgumentParser()
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

class nomalprocess():
	# def __init__(self,mins,maxs):
	# 	#self.mins = mins
	# 	#self.maxs = maxs
	def normal(self,pixels):
		high,wid = pixels.shape
		self.mins = np.min(pixels)
		self.maxs = np.max(pixels)
		for i in range(high):
			for j in range(wid):
				pixels[i][j] = int((pixels[i][j]-self.mins)*(255/(self.maxs-self.mins)))
		return pixels
	def denomal(self,pixels):
		high,wid = pixels.shape
		for i in range(high):
			for j in range(wid):
				pixels[i][j] = int(((self.maxs-self.mins)/255)*pixels[i][j]+self.mins)
		return pixels

top_path = r"D:\python\EDSR-Tensorflow-master\result\experiment4"

outimg_dir = os.path.join(top_path,"x2out_dcm")
imgs = os.listdir(top_path)

if not os.path.exists(outimg_dir):
	os.mkdir(outimg_dir)
for x in imgs:
	imgpath = os.path.join(top_path,x)
	img = pydicom.read_file(imgpath)
	inputs = img.pixel_array
	nopro = nomalprocess()
	inputs = nopro.normal(inputs)
	#misc.imsave(x+'input.jpg',inputs)
	inputs = np.expand_dims(inputs,axis=2)
	outputs = network.predict(inputs)
	outputs = outputs[:,:,0]#for saving dicom
	outputs = nopro.denomal(outputs)
	#misc.imsave(x+'output.jpg',outputs)
	outputs = np.uint16(outputs)
	out_name = 'x2out'+x
	out_img_path = os.path.join(outimg_dir,out_name)
	#for saving dicom
	img.PixelData = outputs.tobytes()
	img.Rows,img.Columns = outputs.shape
	pixel_spacing = img.PixelSpacing
	ps1 = float(pixel_spacing[0])/2
	ps2 = float(pixel_spacing[1])/2
	pixel_spacing = [str(ps1),str(ps2)]
	img.PixelSpacing = pixel_spacing
	img.save_as(out_img_path)
