import scipy.misc
import cv2
import imageio
import random
import numpy as np
import pandas as pd
import os

"""
Load set of images in a directory.
This will automatically allocate a 
random 20% of the images as a test set

data_dir: path to directory containing images
"""


def normal(pixels):
    high, wid = pixels.shape
    min = np.min(pixels)
    max = np.max(pixels)
    for i in range(high):
        for j in range(wid):
            pixels[i][j] = int((pixels[i][j] - min) * (256 / (max - min)))
    return pixels


def load_dataset(data_dir, img_size):
    """img_files = os.listdir(data_dir)
	test_size = int(len(img_files)*0.2)
	test_indices = random.sample(range(len(img_files)),test_size)
	for i in range(len(img_files)):
		#img = scipy.misc.imread(data_dir+img_files[i])
		if i in test_indices:
			test_set.append(data_dir+"/"+img_files[i])
		else:
			train_set.append(data_dir+"/"+img_files[i])
	return"""
    global train_set
    global test_set
    imgs = []
    img_files = os.listdir(data_dir)
    for img in img_files:
        try:
            path = os.path.join(data_dir, img)
            # tmp= pydicom.read_file(path)
            # x = tmp.Rows
            # y = tmp.Columns
            tmp = cv2.imread(path)
            x, y, z = tmp.shape
            coords_x = x // img_size
            coords_y = y // img_size
            coords = [(q, r) for q in range(coords_x) for r in range(coords_y)]

            for coord in coords:
                imgs.append((path, coord))
        except:
            print("oops")
    test_size = int(len(imgs) * 0.2)
    random.shuffle(imgs)
    test_set = imgs[:test_size]
    train_set = imgs[test_size:]
    return


"""
Get test set from the loaded dataset

size (optional): if this argument is chosen,
each element of the test set will be cropped
to the first (size x size) pixels in the image.

returns the test set of your data
"""

"""
Simple method to crop center of image

img: image to crop
cropx: width of crop
cropy: height of crop
returns cropped image
"""


def crop_center(img, cropx, cropy):
    y, x, z_ = img.shape
    startx = random.sample(range(x - cropx - 1), 1)[0]  # x//2-(cropx//2)
    starty = random.sample(range(y - cropy - 1), 1)[0]  # y//2-(cropy//2)
    return img[starty:starty + cropy, startx:startx + cropx]


def my_transforms(lr, hr, crop_size=(128, 128), scale=2):
    """
    Args lr: low resolution Image
         hr: high resolution Image
         crop_size: output size
    return: cropped image alina to the output size
    """
    w, h = lr.size  # get the original size
    tw, th = crop_size

    if w == tw and h == th:
        lr_crop = lr[0:h, 0:w, :]
        hr_crop = hr[0:scale * h, 0:scale * w, :]
        return lr_crop, hr_crop

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    lr_crop = lr[i:th, j:tw, :]
    hr_crop = hr[scale * i:scale * th, scale * j:scale * tw, :] # 2 times crop_size

    return lr_crop, hr_crop


def train_generator(batch_size, img_dir, target_dir, df_train, img_size,
				   scale, shuffle=True):
    df = pd.read_csv(df_train)
    file_lists = df["name"].values
    target_size = img_size * scale
    n_list = len(file_lists)
    while 1:
        if shuffle:
            file_lists = np.random.random_sample(n_list)
        img_batch = np.zeros((batch_size,) + (img_size, img_size, 3))
        target_batch = np.zeros((batch_size,) + (target_size, target_size,3))
        for i in range(n_list // batch_size):
            for j in range(batch_size):
                img = cv2.imread(os.path.join(img_dir, file_lists[i * batch_size + j]))
                target = cv2.imread(os.path.join(target_dir, file_lists[i * batch_size + j]))
                img, target = my_transforms(img, target, crop_size=(img_size, img_size))
                img_batch[j] = img
                target_batch[j] = target
            yield img_batch, target_batch