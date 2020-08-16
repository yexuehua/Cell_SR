import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def load_image(hr_file_path,lr_file_path):
    hr_img = Image.open(hr_file_path)
    lr_img = Image.open(lr_file_path)
    return lr_img,hr_img


class ImageDatasetFromFile(data.Dataset):
    # define the constraction
    def __init__(self, lr_lists, hr_lists):
        super(ImageDatasetFromFile, self).__init__()

        self.lr_lists = lr_lists
        self.hr_lists = hr_lists

        self.input_transform = transforms.Compose([
                                        transforms.RandomCrop(128,128),
                                        transforms.ToTensor()
                                ])


    def __getitem__(self, item):

        lr,hr = load_image(self.lr_lists, self.hr_lists)

        lr_img = self.input_transform(lr)
        hr_target = self.input_transform(hr)

        return lr_img,hr_target


    def __len__(self):
        return len(self.lr_lists)