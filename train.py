import numpy as np
import os, time, sys
from os.path import join as pjoin
import argparse
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from tensorboardX import SummaryWriter

from utils import AverageMeter, RunningScore, Logger
from dataset.dataset import VOC12
import models


def main(args):
    # ================== Setup device, seed and log ============
    np.random.seed(42)
    torch.manual_seed(42)
    if args.cuda: 
        torch.cuda.manual_seed_all(42)
    device = 'cuda' if args.cuda else 'cpu'
    lr = int(-np.log10(args.lr))
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logdir = pjoin(args.save_dir, args.model, f'lr{lr}', f'{args.optim}', f'patience{args.patience}')
    logger = Logger(pjoin(logdir, f'{stamp}.log'))
    logger.write(f'\nTraining configs: {args}')
    writer = SummaryWriter(log_dir=logdir)

    # ===================== Load processed data ====================
    train_dataset = VOC12(args.data_dir, img_size=args.img_size, split='train')
    val_dataset = VOC12(args.data_dir, img_size=args.img_size, split='val')
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size)
    n_classes = val_dataset.n_classes

    # ================== Init model ===================
    model = models.get_model(name=args.model, n_classes=n_classes)
    model = model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # =================== Setup optimizer, scheduler and loss ==========
    if args.optim == 'adam':
        optimizer = Adam([
            {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'], 'lr': 2 * args.lr}, 
            {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'], 'lr': args.lr, 'weight_decay': 2e-5}
        ], betas=(0.95, 0.999))
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=2e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience, min_lr=1e-10)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ======================= Begin training and testing ====================
    start_epoch = 0
    if args.resume:
        logger.write("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint['epoch']
    logger.write(f'start epoch: {start_epoch}, n_train: {len(train_loader.dataset)}, n_val: {len(val_loader.dataset)}')
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    running_metrics_val = RunningScore(n_classes)
    best_iou = -100.0
    # ====================== Training ==========================
    for epoch in range(start_epoch, args.num_epochs):
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
            train_loss_meter.update(loss.item())
            if args.print_freq and (step+1) % args.print_freq == 0:
                logger.write(f'Epoch: [{epoch}][{step+1}/{len(train_loader)}]\t'
                             f'Loss: {loss.item():.4f}\t'
                             f"lr: {optimizer.param_groups[0]['lr']:.9f}\t")
        logger.write(f'Train {epoch} cost: {round(time.time()-st)}s\t'
                     f'Train loss: {train_loss_meter.avg}')
        
        # ======================= validation ==========================
        model.eval()
        st = time.time()
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
        scheduler.step(val_loss_meter.avg)
        logger.write(f'Val {epoch} cost: {round(time.time()-st)}s\t'
                     f'Val loss: {val_loss_meter.avg}')
        
        # ================== Metrics and Save =====================
        score, class_iou = running_metrics_val.get_scores()
        for k, v in score.items():
            logger.write("{}: {}".format(k, v))
            writer.add_scalar("val_metrics/{}".format(k), v, epoch)
        for k, v in class_iou.items():
            logger.write("{}: {}".format(k, v))
            writer.add_scalar("val_metrics/cls_{}".format(k), v, epoch)
        writer.add_scalar("loss/train_loss", train_loss_meter.avg, epoch)
        writer.add_scalar("loss/val_loss", val_loss_meter.avg, epoch)
        train_loss_meter.reset()
        val_loss_meter.reset()
        running_metrics_val.reset()
        if score["Mean IoU"] >= best_iou:
            best_iou = score["Mean IoU"]
            state = {
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_iou": best_iou,
            }
            save_path = pjoin(logdir, f'{stamp}.pkl')
            torch.save(state, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image Segmentation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', required=True)
    parser.add_argument('--save-dir', type=str, default='./saved')
    parser.add_argument('--data-dir', type=str, default='../../dataset/VOCdevkit')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=30)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--patience', type=int, default=100)
    args = parser.parse_args()

    main(args)
