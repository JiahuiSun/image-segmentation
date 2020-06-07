import numpy as np
from os.path import join as pjoin
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VOC12(Dataset):
    """
    root: dir to VOC2012
    img_size: image size after preprocess
    split: data type, train or val
    is_transform:
    """
    def __init__(self, root, img_size=(256, 256), split='train', is_transform=True):
        self.img_dir = pjoin(root, 'JPEGImages')
        self.lab_dir = pjoin(root, 'SegmentationClass')
        self.img_size = img_size
        self.split = split
        self.is_transform = is_transform
        self.data_path = pjoin(root, "ImageSets/Segmentation", split+'.txt')
        with open(self.data_path) as fout:
            self.file_list = [i.rstrip() for i in fout.readlines()]
        self.NUM_CLASSES = 21
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        filename = self.file_list[index]
        img = Image.open(pjoin(self.img_dir, filename+'.jpg'))
        lab = Image.open(pjoin(self.lab_dir, filename+'.png'))
        if self.is_transform:
            img, lab = self.transform(img, lab)
        return img, lab

    def transform(self, img, lab):
        img = img.resize((self.img_size[0], self.img_size[1]))
        lab = lab.resize((self.img_size[0], self.img_size[1]))
        img = self.input_transform(img)
        lab = torch.from_numpy(np.array(lab)).long()
        lab[lab == 255] = 0
        return img, lab

    def __len__(self):
        return len(self.file_list)
