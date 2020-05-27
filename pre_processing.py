import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm


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
files_name = os.listdir("./data/raw")
files_path = [os.path.join("data/raw", i) for i in files_name]

for i in tqdm(range(0, len(files_name)//3)):
    # read a image
    img_1 = cv2.imread(files_path[3*i])
    img_2 = cv2.imread(files_path[3*i+1])
    img_3 = cv2.imread(files_path[3*i+2])

    img_1 = dye(img_1, "blue")
    img_2 = dye(img_2, "green")
    img_3 = dye(img_3, "red")

    img_merge = img_1 + img_2 + img_3
    cv2.imwrite("data/overlay/"+str(i)+"_overlay.tif", img_merge)
# # show the result
# cv2.imshow("merge", img_merge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()