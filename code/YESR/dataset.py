import torch
import random
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image,ImageOps


def load_image(lr_file_path, hr_file_path):
    hr_img = Image.open(hr_file_path).convert(mode="L")
    lr_img = Image.open(lr_file_path).convert(mode="L")
    lr_img = ImageOps.fit(lr_img,size=(512,512),method=Image.NEAREST)
    return lr_img,hr_img


def my_transfroms(lr,hr,crop_size=(128,128),scale=1):
    """
    Args lr: low resolution Image
         hr: high resolution Image
         crop_size: output size
    return: lr = crop_size; hr = double crop_size
    """
    w, h = lr.size #get the original size
    tw, th = crop_size

    if w == tw and h == th:
        lr_crop = F.crop(lr, 0, 0, w, h)
        hr_crop = F.crop(hr, 0, 0, scale*w, scale*h)
        return F.to_tensor(lr_crop), F.to_tensor(hr_crop)

    i = random.randint(0, h-th)
    j = random.randint(0, w-tw)
    lr_crop = F.crop(lr, i, j, th, tw)
    hr_crop = F.crop(hr, scale*i, scale*j, scale*th, scale*tw)

    return F.to_tensor(lr_crop), F.to_tensor(hr_crop)

class ImageDatasetFromFile(data.Dataset):
    # define the constraction
    def __init__(self, lr_lists, hr_lists, root_path):
        super(ImageDatasetFromFile, self).__init__()

        self.lr_lists = lr_lists
        self.hr_lists = hr_lists
        self.root_path = root_path


    def __getitem__(self, item):
        lr_path = join(self.root_path,self.lr_lists[item])
        hr_path = join(self.root_path,self.hr_lists[item])
        lr, hr = load_image(lr_path, hr_path)
        return my_transfroms(lr, hr, crop_size=(100,100), scale=2)


    def __len__(self):
        return len(self.lr_lists)