import pywt
import cv2
import os
import numpy as np
import pywt.data
import matplotlib.pyplot as plt

path = r"C:\Users\212774000\Documents\python\gitclone\SR\Cell_SR\code"
file_name = "merge.png"
# img2 = pywt.data.camera()
img = cv2.imread(os.path.join(path,file_name))
img_b = img[:,:,0]
img_g = img[:,:,1]
img_r = img[:,:,2]
# print(pywt.wavelist(kind='discrete'))
# np_img = np.ndarray((100,100,10))
# print(np_img.shape)
LL,(LH,HL,HH) = pywt.dwt2(img_r,"haar")
merge_b = pywt.dwt2(img_b,"haar")[0]
merge_g = pywt.dwt2(img_g,"haar")[0]
merge_r = pywt.dwt2(img_r,"haar")[0]
img_dst = np.zeros((256,256,3))
img_dst[:,:,0] = merge_b
img_dst[:,:,1] = merge_g
img_dst[:,:,2] = merge_r
print(img_dst.shape)
cv2.imwrite("merge-1.png",img_dst)
#result = pywt.dwtn(img,"haar")
# LL = result[0]
# LL = LL/np.max(LL)*255
# print(result.keys())
# LL = result['aaa']
# LH = result["ada"]
# HL = result["dda"]
# HH = result["ddd"]
fig = plt.figure()
fig.add_subplot(2,2,1)
plt.imshow(LL, cmap="gray")
fig.add_subplot(2,2,2)
# plt.show()
plt.imshow(LH, cmap="gray")
fig.add_subplot(2,2,3)
plt.imshow(HL,cmap="gray")
fig.add_subplot(2,2,4)
plt.imshow(HH,cmap="gray")
# plt.savefig("6_red.png")
plt.show()

# cv2.imshow("LL",LL)
# cv2.imshow("LH",LH)
# cv2.imshow("HL",HL)
# cv2.imshow("HH",HH)
# cv2.waitKey()
