import argparse
import functools
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.datasets import *
from src.model_mixerconv2T42 import Model
from src.losses import loss_qf
from src.utils import AverageMeter, peak_signal_noise_ratio, structural_similarity, fix_random_seeds, perceptual_similarity

import lpips
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Advanced RCT")

# Experiment options
parser.add_argument("--title", type=str, default="Advanced RCT")
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--data_parallel", type=bool, default=False)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument('--device_ids', nargs='+', default=["cuda:0", "cuda:1"])

# Dataset options
parser.add_argument("--root", type=str, default="../data/fiveK")
parser.add_argument("--train_dataset", type=str, default="../data_json/fiveK-train.json")
parser.add_argument("--test_dataset", type=str, default="../data_json/fiveK-test.json")

# Loss options
parser.add_argument('--color_loss', type=float, default = 6e-2)
parser.add_argument('--frequency_loss', type=float, default = 1e-1)
parser.add_argument('--entropy_loss', type=float, default = 2e-1)
parser.add_argument('--frequency_grid', type=int, default = 2)

# Train options
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--lr_min", type=float, default=1e-6)
parser.add_argument("--weight_decay", type=float, default=5e-2)
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--pretrained_model", type=str, default="checkpoints/LIT-LoL-latest.pth")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

parser.add_argument("--write", type=bool, default=False)
parser.add_argument("--write_loss", type=bool, default=False)

def main(args):
    # fix random seed
    fix_random_seeds()

    print(args)

    # build data
    train_dataset = Adobe5kPairedDataset(args.train_dataset, args.root ,training=True)
    test_dataset = Adobe5kPairedDataset(args.test_dataset, args.root ,training=False)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True, 
        drop_last=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size = 1, 
        num_workers=args.num_workers,
        pin_memory=True)
        
    # build model
    model = Model()
    model = model.to(args.device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # build criterion
    criterion = functools.partial(loss_qf, \
                                    scale_cx = args.color_loss, scale_e = args.entropy_loss, \
                                    scale_f = args.frequency_loss, n_grid = args.frequency_grid)

    # build optimizer
    optimizer = AdamW(model.parameters(), args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, len(train_loader) * args.epochs, args.lr_min)

    # load pretrained model
    best_psnr = 0
    best_ssim = 0
    best_lpips = 0
    start_epoch = 0
    logs = []
    if args.resume:
        checkpoint = torch.load(args.pretrained_model)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        best_lpips = checkpoint['best_lpips']
        
        with open(f"{args.checkpoint_dir}/{args.title}-log.txt", "rt") as f:
            logs = f.readlines()
        print(start_epoch, best_psnr)

 # custom loop
    for epoch in range(start_epoch, args.epochs):
        # train
        train_summary = train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch)

        # test
        test_summary = evaluate(test_loader, model)

        if args.write :
            writer.add_scalars('psnr',{'train':train_summary[0], 'test':test_summary[0]},epoch)
        # save
        save_dict = {
            "epoch": epoch + 1,
            "psnr": test_summary[0],
            "ssim": test_summary[1],
            "lpips" : test_summary[2],
            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "best_lpips": best_lpips,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if test_summary[0] > best_psnr:
            best_psnr = test_summary[0]
            best_ssim = test_summary[1]
            best_lpips = test_summary[2]
            save_dict["best_psnr"] = best_psnr
            save_dict["best_ssim"] = best_ssim
            save_dict["best_lpips"] = best_lpips
            checkpoint_path = f"{args.checkpoint_dir}/{args.title}-best.pth"
            torch.save(save_dict, checkpoint_path)
        
        checkpoint_path = f"{args.checkpoint_dir}/{args.title}-latest.pth"
        torch.save(save_dict, checkpoint_path)

        # log
        logs.append([
            "------------------------------------------------------------\n",
            f"Epoch: {epoch+1:03d}\n",
            f"[Train] PSNR:{train_summary[0]:.4f}, SSIM: {train_summary[1]:.4f}, LPIPS: {train_summary[2]:.4f}\n",
            f"[Test] PSNR:{test_summary[0]:.4f}, SSIM: {test_summary[1]:.4f}, LPIPS: {test_summary[2]:.4f}\n",
            f"[Test] Best PSNR: {best_psnr:.4f}, Best SSIM: {best_ssim:.4f}, LPIPS: {best_lpips:.4f}\n",
        ])
        with open(f"{args.checkpoint_dir}/{args.title}-log.txt", "wt") as f:
            for log in logs:
                f.writelines(log)


def train_one_epoch(loader, model, criterion, optimizer, scheduler, epoch):

    model.train()

    epoch_psnr = AverageMeter()
    epoch_ssim = AverageMeter()
    epoch_alex = AverageMeter()


    for i, (inputs, targets) in enumerate(loader):
        # cpu -> gpu
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        # Y, X = model(inputs)
        Y, X, R = model(inputs)
        
        # compute loss and psnr and ssim
        loss, loss_e = criterion(targets, inputs, Y, X, R)
        
        if args.write_loss :
            writer.add_scalar('loss/total loss', loss, epoch*len(loader)+i)
            writer.add_scalar('loss/entropy loss', loss_e, epoch*len(loader) + i)

        # backward and optim step
        loss.backward()
        optimizer.step()

        # update learning rate
        scheduler.step()
        
        # statistics
        batch_size = inputs[0].size(0)
        psnr = peak_signal_noise_ratio(Y, targets)
        ssim = structural_similarity(Y, targets)
        l_alex = perceptual_similarity(lpips_alex, Y, targets)

        epoch_psnr.update(psnr.detach().cpu().numpy(), batch_size)
        epoch_ssim.update(ssim.detach().cpu().numpy(), batch_size)
        epoch_alex.update(l_alex.detach().cpu().numpy(), batch_size)

    return (epoch_psnr.avg, epoch_ssim.avg, epoch_alex.avg)


def evaluate(loader, model):
    
    model.eval()

    epoch_psnr = AverageMeter()
    epoch_ssim = AverageMeter()
    epoch_alex = AverageMeter()

    for inputs, targets in loader: 
        # cpu -> gpu
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        # forward pass
        # Y,_ = model(inputs)
        Y, _,_ = model(inputs)
        
        # statistics
        batch_size = inputs[0].size(0)
        psnr = peak_signal_noise_ratio(Y, targets)
        ssim = structural_similarity(Y, targets)
        l_alex = perceptual_similarity(lpips_alex, Y, targets)

        epoch_psnr.update(psnr.detach().cpu().numpy(), batch_size)
        epoch_ssim.update(ssim.detach().cpu().numpy(), batch_size)
        epoch_alex.update(l_alex.detach().cpu().numpy(), batch_size)
    
    return (epoch_psnr.avg, epoch_ssim.avg, epoch_alex.avg)


if __name__=="__main__":
    args = parser.parse_args()
    lpips_alex = lpips.LPIPS(net='alex').to(args.device)
    if args.write :
        writer = SummaryWriter('runs/'+ args.title)
    main(args)
    if args.write :
        writer.close()
