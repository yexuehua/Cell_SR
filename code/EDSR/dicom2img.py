import pydicom
import cv2
import os
import numpy as np

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

top_path = r"D:\python\EDSR-Tensorflow-master\result\all_scaletest\x2out_dcm"
norm = nomalprocess()
dcm = "x2out6-25.dcm"
dcm_path = os.path.join(top_path,dcm)
dcm_img = pydicom.read_file(dcm_path)
dcm_data = dcm_img.pixel_array
dcm_data = norm.normal(dcm_data)
cv2.imwrite(top_path+"/en6-25.jpg",dcm_data,[int(cv2.IMWRITE_JPEG_QUALITY),100])

# for root,dir,file in os.walk(top_path):
#     if len(file) != 0:
#         for dcm in file:
#             dcm_path = os.path.join(root,dcm)
#             dcm_img = pydicom.read_file(dcm_img)
#             dcm_data = dcm_img.pixel_array
#             cv2.imwrite(r"",dcm_data,[int(cv2.IMWRITE_JPEG_QUALITY),100])

# img = cv2.imread(r'D:\python\EDSR-Tensorflow-master\result\resultmed_dateset\MRdata\mr2\t2\in\in_1.jpg',cv2.IMREAD_GRAYSCALE)
# #img = dcm.pixel_array
# x,y = img.shape
# out = cv2.resize(img,(x*2,y*2),interpolation=cv2.INTER_CUBIC)
# #dcm.Rows,dcm.Columns = out.shape
# #dcm.PixelData = out.tobytes()
# #dcm.save_as("downcubic_x2_1.DCM")
# cv2.imwrite(r'D:\python\EDSR-Tensorflow-master\result\resultmed_dateset\MRdata\mr2\t2\in\cubicup_1.jpg',out,[int(cv2.IMWRITE_JPEG_QUALITY),100])
