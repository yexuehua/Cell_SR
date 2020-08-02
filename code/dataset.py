import torch
import torch.utils.data as data
import torchvision.transforms as transforms


def get_data(item):
    return item


class ImageDatasetFromFile(data.Dataset):
    # define the constraction
    def __init__(self,params):
        super(ImageDatasetFromFile, self).__init__()

        self.params = params

        self.input_transform = transforms.Compose([
                                        transforms.ToTensor()
                                ])


    def __getitem__(self, item):

        data = get_data(item)
        label = get_data(item)

        return data,label


    def __len__(self):
        return len(self.filelists)