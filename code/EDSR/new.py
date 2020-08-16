#import tensorflow as tf
# import imageio
#import scipy.misc
# a = tf.constant(2)
# b = tf.constant(4)
# c = a + b
# d = imageio.imread(r'D:\python\EDSR-Tensorflow-master\dataset\2092.jpg')
# print(d)
# print(d.shape)
# sess = tf.Session()
# print(sess.run(c))
# import os
# import pydicom
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
'''
x = pydicom.read_file(r"D:\python\EDSR-Tensorflow-master\med_dateset\CTdata\CT1\1.2.392.200036.9116.2.1220972159.1407127612.8906.1.20.dcm")
#x = pydicom.read_file(r"D:\python\EDSR-Tensorflow-master\med_dateset\MRdata\mr1\t1ce\1.DCM")

a = x.pixel_array

high,wid = a.shape
print(high)
for i in range(high):
    for j in range(wid):
        a[i][j] = int((a[i][j]+2048)*(256/4096))
#a = cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)
a = np.expand_dims(a, axis=2)
a = np.concatenate((a, a, a), axis=-1)
b = np.uint8(a)
print(b.dtype)
print(b.shape)
plt.imshow(a)
plt.show()



def normal(pixels):
	high,wid = pixels.shape
	for i in range(high):
		for j in range(wid):
			pixels[i][j] = int((pixels[i][j]+2048)*(256/4096))
	return pixels

def denormal(pixels):
	high,wid = pixels.shape
	for i in high:
		for j in wid:
			pixels[i][j] = int((pixels[i][j]-127)*(2048/127))
	return pixels

'''

# import pydicom
# import numpy as np
# a = pydicom.read_file(r'D:\yexuehua\data\JIN_CHEN\NAC1_delay\10-19.dcm')
# b = a.pixel_array
# c = np.min(b)
# d = np.max(b)
#
# #a.PixelSpacing = ['0.63','0.63']
#
# print(a)
# print(b.dtype)
# print(c)
# print(d)

# import cv2
# import numpy as np
# a = cv2.imread('lion.png',cv2.IMREAD_GRAYSCALE)
# b = np.expand_dims(a,axis=2)
# c = cv2.resize(b,(500,500))
# print(b.shape,'\n',c.shape)
import cv2
from scipy import misc
import numpy as np
a = cv2.imread('lion.png',cv2.IMREAD_GRAYSCALE)
b = np.expand_dims(a,axis=2)
c = misc.imresize(b,(500,500))
print(c.shape)





