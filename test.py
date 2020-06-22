import numpy as np
import torch
import os, time, sys
from os.path import join as pjoin
from PIL import Image
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

import models
from utils import convert_state_dict, Logger
from dataset.dataset import VOC12


def main(args):
    # ========= Setup device and seed ============
    np.random.seed(42)
    torch.manual_seed(42)
    if args.cuda:
        torch.cuda.manual_seed_all(42)
    device = 'cuda' if args.cuda else 'cpu'
    logger = Logger(pjoin(args.save_dir, args.model, 'test.log'))
    logger.write(f'\nTesting configs: {args}')

    # ================= Load processed data ===================
    val_dataset = VOC12(args.data_dir, img_size=args.img_size, split='test')
    val_loader = DataLoader(val_dataset, num_workers=8, batch_size=1)
    n_classes = val_dataset.n_classes

    # ================= Init model ====================
    model = models.get_model(name=args.model, n_classes=n_classes)
    model = model.to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()

    # ====================== Only one image ==========================
    if args.eval:
        with torch.no_grad():
            img = Image.open(args.img_path)
            origin = img.size
            if args.img_size:
                img = img.resize((val_dataset.img_size[0], val_dataset.img_size[1]))
            img = val_dataset.input_transform(img).unsqueeze(0).to(device)
            out = model(img)
            pred = np.squeeze(out.data.max(1)[1].cpu().numpy(), axis=0)
            decoded = val_dataset.decode_segmap(pred)
            img_out = ToPILImage()(decoded).resize(origin)
            img_out.save(pjoin(args.save_dir, args.model, f'eval_{args.img_size}.png'))
        return

    # ====================== Testing Many images ==============================
    with torch.no_grad():
        for idx, (name, img) in enumerate(val_loader):
            img = img.to(device)
            out = model(img)
            pred = out.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            decoded = val_dataset.decode_segmap(pred)
            ToPILImage()(decoded).save(pjoin(args.save_dir, args.model, f'{name[0]}_{args.img_size}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Segmentation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', type=str, default='fcn8')
    parser.add_argument('--data-dir', type=str, default='/home/jinHM/sunjiahui/MachineLearning/dataset/VOCdevkit')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--model-path', type=str, default='./saved')
    parser.add_argument('--img-path', type=str, default='./visual/2007_000129.jpg')
    parser.add_argument('--save-dir', type=str, default='./saved')

    parser.add_argument('--img-size', type=int, default=256)
    args = parser.parse_args()

    main(args)
