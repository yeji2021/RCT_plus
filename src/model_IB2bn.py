import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.autograd import Variable

from timm.models.layers import DropPath

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.globalRCT = GlobalRCT()
        self.initialize_weights()

    def forward(self, x):
        x = x.type(torch.float32)
        # x.shape : [batch, C, H, W]
        Z = self.encoder(x) # ZG shape : [B, 128, 1, 1]
        B, C, H, W = Z.size()
        F = Z.reshape([B,C, H*W])
        T1, T2, R = self.globalRCT(Z)
        Y = matrix(F,R,T1) 
        X = matrix(F,R,T2)
        # reshape [batch, HW, 3] -> [batch, 3, H, W]
        Y = Y.permute(0,2,1).reshape(-1,3,H,W)
        X = X.permute(0,2,1).reshape(-1,3,H,W)
        return Y, X, R

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


def matrix(f,r,t):
    # f : B x C x H*W, r : B x C x N
    C = f.shape[1]
    f_t = f.permute(0,2,1)
    # batch-wise dot product : torch.bmm == torch.matmul
    softmax = nn.Softmax(dim = -1)
    tanh = nn.Tanh()
    A = softmax(torch.bmm(f_t, r)/torch.sqrt(torch.tensor(C))) # B x HW x N
    T_t = t.permute(0,2,1) # B x N x 3
    Y = tanh(torch.bmm(A, T_t)) # B x HW x 3 in [-1,1]
    
    return Y

class GlobalRCT(nn.Module):
    def __init__(self):
        
        super().__init__()
        # Z : Bx128x1x1

        self.Representative = nn.Sequential(
            DepthConvBlock(16,32,3,2,1),
            DepthConvBlock(32,64,3,2,1),
            DepthConvBlock(64,128,3,2,1),
            DepthConvBlock(128,256,3,2,1),
            DepthConvBlock(256,1024,3,2,1),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.TransformedColor1 = nn.Sequential(
            DepthConvBlock(16,32,3,2,1),
            DepthConvBlock(32,64,3,2,1),
            DepthConvBlock(64,3*64,3,2,1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(3*64, 3*64)
        )

        self.TransformedColor2 = nn.Sequential(
            DepthConvBlock(16,32,3,2,1),
            DepthConvBlock(32,64,3,2,1),
            DepthConvBlock(64,3*64,3,2,1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(3*64, 3*64)
        )


    def forward(self, z,):
        z = TF.resize(z, (256,256)).type(torch.float32)

        R = self.Representative(z) 
        R = R.reshape([-1, 16, 64]) # [B, 16, 64]

        T1 = self.TransformedColor1(z) # [B, 192, 1, 1]
        T1 = T1.reshape([-1, 3, 64]) # [B, 3, 64]

        T2  = self.TransformedColor2(z)
        T2 = T2.reshape([-1, 3, 64])

        return T1, T2, R


class Encoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.InvertedBottle = InvertedBottle(3, 16, 5, 1, 2,) # 3- 12 - 16
    
    def forward(self, x):
        # stage 1
        f4 = self.InvertedBottle(x)

        return f4

class DepthConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0):
        super(DepthConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size, stride, padding, groups = in_dim)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 1)
        self.norm1 = nn.BatchNorm2d(in_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x) 
        x = self.conv2(x)
        return x



class InvertedBottle(nn.Module):
    def __init__(self, in_dim, out_dim, kernel = 3, stride = 1, padding = 1, expansion = 4):
        super().__init__()
        self.h_dim = expansion*in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.h_dim, 1),
            nn.BatchNorm2d(self.h_dim),
            nn.SiLU(),
            nn.Conv2d(self.h_dim, self.h_dim, kernel, stride, padding, groups=self.h_dim),
            nn.BatchNorm2d(self.h_dim),
            nn.SiLU(),
            nn.Conv2d(self.h_dim, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim) 
        )
        self.projection = nn.Conv2d(self.in_dim, self.out_dim, 1, stride, 0)

    def forward(self, x):
        Fx = self.conv(x)
        if self.in_dim != self.out_dim :
            wx = self.projection(x)
            y = Fx + wx 
        else :
            y = Fx + x
        return y