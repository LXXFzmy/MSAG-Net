import torch.utils.data as data
import os.path
import torchvision.transforms as T
from PIL import Image


def make_dataset(dir):
    img = os.listdir(dir)
    number_sample = len(img)
    return img, number_sample

def pil_loader(path):

    img = Image.open(path)
    
    return img.convert("RGB")

def pil_loader0(path):

    img = Image.open(path)

    #return img.convert("L")
    return img

class Imageio(data.Dataset):
    def __init__(self, root0, root1, loader=pil_loader):
        img0, number0 = make_dataset(root0)
        img1, number1 = make_dataset(root1)

        self.number_of_samples = number0

        self.img0 = img0
        self.img1 = img1

        self.root0 = root0
        self.root1 = root1

    def __getitem__(self, index):

        path0 = os.path.join(self.root0, self.img0[index])
        path1 = os.path.join(self.root1, self.img1[index])

        img = pil_loader(path0)
        gt = pil_loader0(path1)

        img = T.ToTensor()(img)
        img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        gt = T.ToTensor()(gt)

        return [img, gt]

    def __len__(self):
        return self.number_of_samples


class Imageio_test(data.Dataset):
    def __init__(self, root0,  loader=pil_loader):
        img0, number0 = make_dataset(root0)
        self.number_of_samples = number0
        self.img0 = img0
        self.root0 = root0
        self.loader = loader

    def __getitem__(self, index):
        path0 = os.path.join(self.root0, self.img0[index])

        img = self.loader(path0)
        img = T.Resize((480, 640))(img)
        img = T.ToTensor()(img)
        img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return [img, self.img0[index]]

    def __len__(self):
        return self.number_of_samples