import cv2
import pydicom
import os
import numpy as np
from tqdm import tqdm
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

norm = nomalprocess()

top_path = r"D:\python\EDSR-Tensorflow-master\result\all_scaletest\opening_report"

outimg_dir = os.path.join(top_path,"x2cubic_out_dcm")
imgs = os.listdir(top_path)

if not os.path.exists(outimg_dir):
	os.mkdir(outimg_dir)
for x in tqdm(imgs):
	imgpath = os.path.join(top_path,x)
	img = pydicom.read_file(imgpath)
	inputs = img.pixel_array
	nopro = nomalprocess()
	inputs = nopro.normal(inputs)
	#misc.imsave(x+'input.jpg',inputs)
	inputs = np.expand_dims(inputs,axis=2)
	h,w,_= inputs.shape
	out = cv2.resize(inputs,(h*2,w*2),interpolation=cv2.INTER_LINEAR)

	outputs = out[:,:]#for saving dicom
	outputs = nopro.denomal(outputs)

	#misc.imsave(x+'output.jpg',outputs)
	outputs = np.uint16(outputs)
	out_name = 'x2cubic_out'+x
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



#
# #dcm = pydicom.read_file(r'D:\python\EDSR-Tensorflow-master\Experiment2\x2out_dcm\x2out1.DCM')
# img = cv2.imread(r'D:\python\EDSR-Tensorflow-master\result\resultmed_dateset\MRdata\mr2\t2\in\in_1.jpg',cv2.IMREAD_GRAYSCALE)
# #img = dcm.pixel_array
# x,y = img.shape
# out = cv2.resize(img,(x*2,y*2),interpolation=cv2.INTER_CUBIC)
# #dcm.Rows,dcm.Columns = out.shape
# #dcm.PixelData = out.tobytes()
# #dcm.save_as("downcubic_x2_1.DCM")
# cv2.imwrite(r'D:\python\EDSR-Tensorflow-master\result\resultmed_dateset\MRdata\mr2\t2\in\cubicup_1.jpg',out,[int(cv2.IMWRITE_JPEG_QUALITY),100])
