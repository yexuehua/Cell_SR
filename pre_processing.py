import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def dye(img, color="red"):
    # get index of channel
    idx = color_dict[color]
    rgb_list = [0,1,2]
    rgb_list.remove(idx)

    # normalize the image
    img[:, :, idx] = img[:, :, idx]/np.max(img[:, :, idx])*255
    # img[:, :, idx] = cv2.equalizeHist(img[:, :, idx])

    img[:, :, rgb_list[0]] = 0
    img[:, :, rgb_list[1]] = 0

    return img


color_dict = {"red": 2, "green": 1, "blue": 0}
# get the path of each images
files_name = os.listdir("./data")
files_path = [os.path.join("data", i) for i in files_name]

# read a image
img_1 = cv2.imread(files_path[0])
img_2 = cv2.imread(files_path[1])

img_1 = dye(img_1, "blue")
img_2 = dye(img_2,"green")

img_merge = img_1 + img_2
# show the result
cv2.imshow("a", img_merge)
cv2.imshow("b", img_1)
cv2.imshow("c", img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()