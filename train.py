import numpy as np
import torch
import os, time, sys
from os.path import join as pjoin
from PIL import Image
import argparse
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import ToPILImage

from utils import decode_segmap, AverageMeter, RunningScore, Logger
from piwise.dataset import VOC12
from piwise.network import FCN8, FCN16, FCN32, UNet, PSPNet, SegNet, ZZ16, fcn16s, fcn8s
from piwise.criterion import CrossEntropyLoss2d, cross_entropy2d


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
            print(f'Epoch: [{epoch}][{step+1}/{len(train_loader)}]\t'
                  f'Loss: {loss.item():.4f}\t'
                  f"lr: {optimizer.param_groups[0]['lr']:.6f}\t")
    print(f'Epoch {epoch} cost: {round(time.time()-st)}s')


def main(args):
    # ========= Setup device and seed ============
    np.random.seed(42)
    torch.manual_seed(42)
    if args.cuda:
        torch.cuda.manual_seed_all(42)
        device = 'cuda'
    else:
        device = 'cpu'
    # ========= Load processed data ============
    train_dataset = VOC12(args.data_dir, img_size=args.img_size, split='trainval')
    val_dataset = VOC12(args.data_dir, img_size=args.img_size, split='val')
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size)
    n_classes = val_dataset.NUM_CLASSES
    # ================== Init model ===================
    if args.model == 'fcn8s':
        vgg16 = models.vgg16(pretrained=True)
        model = fcn8s(n_classes)
        model.init_vgg16_params(vgg16)
    if args.model == 'fcn16s':
        vgg16 = models.vgg16(pretrained=True)
        model = fcn16s(n_classes)
        model.init_vgg16_params(vgg16)
    if args.model == 'fcn32s':
        vgg16 = models.vgg16(pretrained=True)
        model = fcn32s(n_classes)
        model.init_vgg16_params(vgg16)
    if args.model == 'unet':
        model = UNet(n_classes)
    if args.model == 'pspnet':
        model = PSPNet(n_classes)
    if args.model == 'segnet':
        model = SegNet(n_classes)
    if args.model == 'zz16':
        model = ZZ16(n_classes)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # ========= Setup optimizer, scheduler and loss ==========
    # optimizer = Adam(model.parameters(), lr=args.lr)
    # SGD(model.parameters(), 1e-4, .9, 2e-5)
    optimizer = SGD(model.parameters(), lr=1e-5, momentum=0.99, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)
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
    # TODO: meaning of metrics
    val_loss_meter = AverageMeter()
    running_metrics_val = RunningScore(train_dataset.NUM_CLASSES)
    best_iou = -100.0
    for epoch in range(start_epoch, args.num_epochs):
        train(epoch, model, train_loader, criterion, optimizer, device)
        scheduler.step()
        # =========== validation =================
        model.eval()
        with torch.no_grad():
            for step, (img, lab) in enumerate(val_loader):
                img = img.to(device)
                lab = lab.to(device)
                out = model(img)
                val_loss = criterion(out, lab)
                pred = out.data.max(1)[1].cpu().numpy()
                gt = lab.data.cpu().numpy()
                running_metrics_val.update(gt, pred)
                val_loss_meter.update(val_loss.item())
        score, class_iou = running_metrics_val.get_scores()
        for k, v in score.items():
            print("{}: {}".format(k, v))
        # for k, v in class_iou.items():
        #     print("{}: {}".format(k, v))
        val_loss_meter.reset()
        running_metrics_val.reset()
        if score["Mean IoU : \t"] >= best_iou:
            best_iou = score["Mean IoU : \t"]
            state = {
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_iou": best_iou,
            }
            save_path = pjoin(args.save_dir, f"{args.model}_best_model.pkl")
            torch.save(state, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Segmentation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', type=str, default='zz16')
    parser.add_argument('--save-dir', type=str, default='./saved')
    parser.add_argument('--data-dir', type=str, default='/home/jjou/sunjiahui/MLproject/dataset/VOCdevkit/VOC2012')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--lr', type=float,  default=0.001)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--print-freq', type=int, default=30)
    args = parser.parse_args()

    sys.stdout = Logger(pjoin(args.save_dir, f'{args.model}_train.log'))
    main(args)
