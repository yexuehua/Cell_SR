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


#for jpg super resolution
'''

inputs = cv2.imread('3.jpg',cv2.IMREAD_COLOR)
outputs = network.predict(inputs)
cv2.imwrite('out_3.jpg',outputs,[int(cv2.IMWRITE_JPEG_QUALITY),100])

'''



#for CT Super resolution
'''
def normal(pixels):
	high,wid = pixels.shape
	for i in range(high):
		for j in range(wid):
			pixels[i][j] = int((pixels[i][j]+2048)*(256/4096))
	return pixels

top_path = r"D:\python\EDSR-Tensorflow-master\med_dateset\CTdata"
mods = os.listdir(top_path)
for mod in mods:
	datapath = os.path.join(top_path,mod)
	inimg_dir = os.path.join(datapath,"in")
	outimg_dir = os.path.join(datapath,"out")
	imgs = os.listdir(datapath)
	if not os.path.exists(inimg_dir):
		os.mkdir(inimg_dir)
	if not os.path.exists(outimg_dir):
		os.mkdir(outimg_dir)
	for x in imgs:
		imgpath = os.path.join(datapath,x)
		img = pydicom.read_file(imgpath)
		inputs = img.pixel_array
		inputs = normal(inputs)
		inputs_img = np.uint8(inputs)
		inputs_img = cv2.cvtColor(inputs_img,cv2.COLOR_GRAY2BGR)
		# inputs = np.expand_dims(inputs, axis=2)
		# inputs = np.concatenate((inputs, inputs, inputs), axis=-1)
		outputs = network.predict(inputs_img)
		x = x.replace('dcm','jpg')
		in_name = "in_" + x
		out_name = "out_" + x
		in_img_path = os.path.join(inimg_dir,in_name)
		out_img_path = os.path.join(outimg_dir,out_name)
		cv2.imwrite(in_img_path,inputs_img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
		cv2.imwrite(out_img_path,outputs,[int(cv2.IMWRITE_JPEG_QUALITY),100])
	
'''

#for MRI Super resolution

top_path = r"D:\python\EDSR-Tensorflow-master\result\experiment3"
#inimg_dir = os.path.join(datapath,"in")
outimg_dir = os.path.join(top_path,"x2out_dcm")
imgs = os.listdir(top_path)
# if not os.path.exists(inimg_dir):
# 	os.mkdir(inimg_dir)
if not os.path.exists(outimg_dir):
	os.mkdir(outimg_dir)
for x in imgs:
	imgpath = os.path.join(top_path,x)
	img = pydicom.read_file(imgpath)
	inputs = img.pixel_array
	inputs= cv2.cvtColor(inputs,cv2.COLOR_GRAY2BGR)
	outputs = network.predict(inputs)
	outputs=outputs[:,:,0]#for saving dicom
	outputs = np.uint8(outputs)#for saving dicom
	#x = x.replace('DCM','jpg')
	#in_name = "in_" + x
	#out_name = 'x2out'+x
	out_name = x.replace('x2','x4')
	#in_img_path = os.path.join(inimg_dir,in_name)
	out_img_path = os.path.join(outimg_dir,out_name)
	#cv2.imwrite(in_img_path,inputs,[int(cv2.IMWRITE_JPEG_QUALITY),100])
	#cv2.imwrite(out_img_path,outputs,[int(cv2.IMWRITE_JPEG_QUALITY),100])

	#for saving dicom

	img.PixelData = outputs.tobytes()
	img.Rows,img.Columns = outputs.shape
	pixel_spacing = img.PixelSpacing
	ps1 = float(pixel_spacing[0])/2
	ps2 = float(pixel_spacing[1])/2
	pixel_spacing = [str(ps1),str(ps2)]
	img.PixelSpacing = pixel_spacing
	img.save_as(out_img_path)





# top_path = r"D:\python\EDSR-Tensorflow-master\Experiment"
# mods = os.listdir(top_path)
# for mod in mods:
# 	datapath = os.path.join(top_path,mod)
# 	#inimg_dir = os.path.join(datapath,"in")
# 	outimg_dir = os.path.join(datapath,"x2out_dcm")
# 	imgs = os.listdir(datapath)
# 	# if not os.path.exists(inimg_dir):
# 	# 	os.mkdir(inimg_dir)
# 	if not os.path.exists(outimg_dir):
# 		os.mkdir(outimg_dir)
# 	for x in imgs:
# 		imgpath = os.path.join(datapath,x)
# 		img = pydicom.read_file(imgpath)
# 		inputs = img.pixel_array
# 		inputs= cv2.cvtColor(inputs,cv2.COLOR_GRAY2BGR)
# 		outputs = network.predict(inputs)
# 		outputs=outputs[:,:,0]#for saving dicom
# 		outputs = np.uint8(outputs)#for saving dicom
# 		#x = x.replace('DCM','jpg')
# 		#in_name = "in_" + x
# 		out_name = "x2out_" + x
# 		#in_img_path = os.path.join(inimg_dir,in_name)
# 		out_img_path = os.path.join(outimg_dir,out_name)
# 		#cv2.imwrite(in_img_path,inputs,[int(cv2.IMWRITE_JPEG_QUALITY),100])
# 		#cv2.imwrite(out_img_path,outputs,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#
# 		#for saving dicom
#
# 		img.PixelData = outputs.tobytes()
# 		img.Rows,img.Columns = outputs.shape
# 		pixel_spacing = img.PixelSpacing
# 		ps1 = float(pixel_spacing[0])/2
# 		ps2 = float(pixel_spacing[1])/2
# 		pixel_spacing = [str(ps1),str(ps2)]
# 		img.PixelSpacing = pixel_spacing
# 		img.save_as(out_img_path)



'''


x = pydicom.read_file('1.2.392.200036.9116.2.1220972159.1407127612.8906.1.26.dcm')
inputs = x.pixel_array
inputs = np.expand_dims(inputs, axis=2)
inputs = np.concatenate((inputs, inputs, inputs), axis=-1)
print(inputs)
outputs = network.predict(inputs)
outputs=outputs[:,:,0]
outputs = np.int16(outputs)
print(outputs)
x.PixelData = outputs.tobytes()
x.Rows,x.Columns = outputs.shape
print(x.pixel_array.shape)
x.save_as('out_1.2.392.200036.9116.2.1220972159.1407127612.8906.1.26.dcm')
#if args.image:
#	scipy.misc.imsave('in16.bmp',inputs)
#	scipy.misc.imsave('out16.bmp',outputs)


'''
