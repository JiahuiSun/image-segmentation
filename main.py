import numpy as np
import torch
import os, time
from PIL import Image
import argparse
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from utils import decode_segmap, AverageMeter
from piwise.dataset import VOC12
from piwise.network import FCN8, FCN16, FCN32, UNet, PSPNet, SegNet, ZZ16
from piwise.criterion import CrossEntropyLoss2d, cross_entropy2d


parser = argparse.ArgumentParser('Image Segmentation')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model', required=True)

subparsers = parser.add_subparsers(dest='mode')
subparsers.required = True

parser_eval = subparsers.add_parser('eval')
parser_eval.add_argument('image')
parser_eval.add_argument('label')

parser_train = subparsers.add_parser('train')
parser_train.add_argument('--data-dir', required=True)
parser_train.add_argument('--lr', type=float,  default=0.001)
parser_train.add_argument('--resume', type=str, default=None)
parser_train.add_argument('--num-epochs', type=int, default=30)
parser_train.add_argument('--num-workers', type=int, default=4)
parser_train.add_argument('--batch-size', type=int, default=16)
parser_train.add_argument('--print-freq', type=int, default=50)
args = parser.parse_args()


def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    st = time.time()
    for step, (img, lab) in enumerate(train_loader):
        img = img.to(device)
        lab = lab.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, lab)
        loss.backward()
        optimizer.step()
        if (step+1) % args.print_freq == 0:
            print(f'Epoch: [{epoch+1}][{step+1}/{len(train_loader)}]\t'
                  f'Loss: {loss.item():.4f}\t')
    print(f'Epoch {epoch} cost: {round(time.time()-st)}s')


# def test(model, device):


def evaluate(model, device):
    model.eval()
    image = input_transform(Image.open(args.image)).unsqueeze(0)
    out = model(image)
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    np.save('rgb.npy', rgb)
    image_transform = ToPILImage()
    image_transform(rgb).save(args.label)


def main():
    # ========= Setup device and seed ============
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        device = 'cuda'
    else:
        device = 'cpu'

    # ========= Init model ==========
    if args.model == 'fcn8':
        Net = FCN8
    if args.model == 'fcn16':
        Net = FCN16
    if args.model == 'fcn32':
        Net = FCN32
    if args.model == 'fcn32':
        Net = FCN32
    if args.model == 'unet':
        Net = UNet
    if args.model == 'pspnet':
        Net = PSPNet
    if args.model == 'segnet':
        Net = SegNet
    if args.model == 'zz16':
        Net = ZZ16
    assert Net is not None, f'model {args.model} not available'
    model = Net(21).to(device)

    # ========= Load processed data ============
    train_dataset = VOC12(args.data_dir, split='train')
    val_dataset = VOC12(args.data_dir, split='val')
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    # ========= Setup optimizer, scheduler and loss ==========
    # optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)
    if args.model.startswith('FCN'):
        optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)
    if args.model.startswith('PSP'):
        optimizer = SGD(model.parameters(), 1e-2, .9, 1e-4)
    if args.model.startswith('Seg'):
        optimizer = SGD(model.parameters(), 1e-3, .9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
    criterion = cross_entropy2d

    # ============= Begin training ==============
    start_epoch = 0
    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint['epoch']
    if args.mode == 'eval':
        evaluate(model, device)
        return
    for epoch in range(start_epoch, args.num_epochs):
        train(epoch, model, train_loader, criterion, optimizer, device)
        scheduler.step()

    # ============= Begin testing ==============
    # test(model, val_loader, device)


if __name__ == '__main__':
    main()
