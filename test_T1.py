# test code ; the model will be create just one T feature. ( for Y tilde )

import argparse
import functools
import signal

from pkg_resources import load_entry_point
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.datasets import *
from src.model_mixerconv2T42_test import Model
from src.utils import AverageMeter, peak_signal_noise_ratio, structural_similarity, fix_random_seeds

import lpips
import cv2
import os
from time import time
import math 
from skimage.metrics import structural_similarity as sk_ssim

parser = argparse.ArgumentParser(description="LIT-LoL")
parser.add_argument("--device", type=str, default="cuda:1")

# Dataset options
parser.add_argument("--root", type=str, default="../data/fiveK")
parser.add_argument("--test_dataset", type=str, default="../data_json/fiveK-test.json")

# Train options
parser.add_argument("--pretrained_model", type=str, default="checkpoints/LIT-LoL-latest.pth")
parser.add_argument("--title", type = str)

parser.add_argument('--batch', type = int, default=1)
parser.add_argument('--save_pred', type= bool, default = False)

def main(args):
    # fix random seed
    fix_random_seeds()

    print(args)
    # build data
    test_dataset = Adobe5kPairedDataset(args.test_dataset, args.root,training=False)
    test_loader = DataLoader(test_dataset, args.batch)

    # build model
    model = Model()
    model = model.to(args.device)

    # load pretrained model
    checkpoint = torch.load(args.pretrained_model, map_location = args.device)
    model.load_state_dict(checkpoint["model"], strict = False)

    start_epoch = checkpoint["epoch"]
    best_psnr = checkpoint["best_psnr"]
    best_ssim = checkpoint["best_ssim"]
    best_lpips = checkpoint['best_lpips']
    print(start_epoch, best_psnr, best_ssim, best_lpips)

    # test
    evaluate(test_loader, model,)


def save_img(output, save_dir, name):
    output = output.squeeze(0).detach().cpu().numpy().transpose(1,2,0)
    #denormalize
    output = (output+1)*127.5

    #rgb2bgr
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_dir + name ,output)
    print(save_dir + name)
    
def evaluate(loader, model,):
    
    model.eval()

    psnr =0
    ssim =0
    alex =0
    t = 0
    s_ssim = 0

    if args.save_pred :
        save_dir = './results/' + args.title + '/'

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    for i, (inputs, targets) in enumerate(loader): 
        # cpu -> gpu
        _,_,h,w = inputs.shape
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        # forward pass
        tmp = time()
        outputs, _ = model(inputs)
        if i != 0:
            t += (time() - tmp)

        # statistics
        psnr += peak_signal_noise_ratio(outputs, targets).item() #[-1,1]
        ssim += structural_similarity(outputs, targets).item()
        alex += loss_fn_alex(outputs, targets).item()

        
        s_ssim += cal_sk_ssim(targets, outputs)
        # print(f' ssim {structural_similarity(outputs, targets).item():.4f}, nu_ssim {cal_ssim(outputs, targets, as_loss = False).item():.4f}')

        # print(f'id {i} psnr {peak_signal_noise_ratio(outputs, targets).item():.4f} \
        # ssim {structural_similarity(outputs, targets).item():.4f} \
        # vgg {loss_fn_alex(outputs, targets).item():.4f}')

        i+=1

    psnr /= len(loader)
    ssim /= len(loader)
    s_ssim /= len(loader)
    alex /= len(loader)
    t /= (len(loader)-1)

    print(f'psnr {psnr:.4f} ssim {ssim:.4f} alex {alex:.4f} time {1000*t:.2f}')
    print(f'sk_ssim {s_ssim:.4f}')

def cal_sk_ssim(targets, outputs):
    # calculate ssim on YCRCB color, using scipy
    outputs = (outputs.squeeze(0).permute(1,2,0).detach().cpu().numpy() + 1 )*127.5 # [0,2], [1,c,h,w]
    outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2YCR_CB)
    o_range = outputs.max() - outputs.min()

    targets = (targets.squeeze(0).permute(1,2,0).detach().cpu().numpy() + 1 )*127.5
    targets = cv2.cvtColor(targets, cv2.COLOR_RGB2YCR_CB)
    t_range = targets.max() - targets.min()

    range = o_range if o_range > t_range else t_range

    sk_s = sk_ssim(outputs,targets,win_size = 11, channel_axis = 2, data_range=range) 
    return sk_s



if __name__=="__main__":
    args = parser.parse_args()
    loss_fn_alex = lpips.LPIPS(net='alex').to(args.device)
    main(args)
