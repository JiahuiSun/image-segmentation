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
    def __init__(self, root, img_size=256, split='train', is_transform=True):
        self.img_dir = pjoin(root, 'JPEGImages')
        self.lab_dir = pjoin(root, 'SegmentationClass')
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
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
        if self.img_size == ("same", "same"):
            pass
        else:
            img = img.resize((self.img_size[0], self.img_size[1]))
            lab = lab.resize((self.img_size[0], self.img_size[1]))
        img = self.input_transform(img)
        lab = torch.from_numpy(np.array(lab)).long()
        lab[lab == 255] = 0
        return img, lab
    
    def decode_segmap(self, image):
        label_colors = np.array([(0, 0, 0),  # 0=background
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        for l in range(0, self.NUM_CLASSES):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
            rgb = np.stack([r, g, b], axis=2)
        return rgb

    def __len__(self):
        return len(self.file_list)
