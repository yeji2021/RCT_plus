import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.globalRCT = GlobalRCT()
        self.initialize_weights()

    def forward(self, x):
        x = x.type(torch.float32)
        # x.shape : [batch, C, H, W]
        F = self.encoder(x) # ZG shape : [B, 128, 1, 1]
        B, C, H, W = F.size()
        image_F = F.reshape([B,C, H*W])
        T1, T2, R, Z = self.globalRCT(F)

        Y = matrix(image_F,R,T1) 
        X = matrix(image_F,R,T2)

        # reshape [batch, HW, 3] -> [batch, 3, H, W]
        Y = Y.permute(0,2,1).reshape(-1,3,H,W)
        X = X.permute(0,2,1).reshape(-1,3,H,W)
        return Y, X, Z

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
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
        )
        self.Rmlp = RLinear()

        self.TransformedColor1 = Tblock()

        self.TransformedColor2 = Tblock()


    def forward(self, x,):
        x = TF.resize(x, (256,256)).type(torch.float32)

        Z = self.Representative(x) # [B, 256, 16, 16]
        B,C,H,W = Z.shape
        Z = Z.reshape([B, C, H*W]) # [B, 256, 256]
        R = self.Rmlp(Z) # R [B, S 64, C 16], F [B, S 64, C 64]
        R = R.permute(0,2,1) # [B, 16, 64]

        T1 = self.TransformedColor1(Z) # [B, S 64, C 3]
        T1 = T1.permute(0,2,1)
        T2  = self.TransformedColor2(Z)
        T2 = T2.permute(0,2,1)

        return T1, T2, R, Z

class RLinear(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.stage1 = LinearDepthBlock(256, 128,3,2,1) 
        self.stage2 = LinearDepthBlock(128, 64,3,2,1) 
        self.last = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        # x shape = [B, C, S]
        x = self.stage1(x) # [B, 128, 128]
        x = self.stage2(x) # [B, 64, 64] [B, C, S]
        R = self.last(x) # [B, 64, 16] 

        return R

class mixer(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=1):
        super().__init__()
        self.h_dim = int(expansion*in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.tokenmlp = nn.Sequential(
            nn.Linear(self.in_dim, self.h_dim),
            nn.SiLU(),
            nn.Linear(self.h_dim, self.out_dim)
        )
        self.channelmlp = nn.Sequential(
            nn.Linear(self.in_dim, self.h_dim),
            nn.SiLU(),
            nn.Linear(self.h_dim, self.out_dim)
        )
        self.layernorm = nn.LayerNorm(in_dim)

    def forward(self, x):
        # x shape [B, S, C]
        x = self.layernorm(x) 

        # token mixing
        y = x.permute(0,2,1) # [B, C, S]
        y = self.tokenmlp(y)
        x = y.permute(0,2,1) # [B, S, C]

        # channel mixing
        y = self.layernorm(x)
        y = self.channelmlp(y) # [B, S, C]

        return y
    
class Tblock(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mixer = mixer(256, 64, expansion=0.5)
        self.last = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16,3)
        )

    def forward(self, x):
        # x shape = [B, S, C]
        x = self.mixer(x) # [B, S 64, C 64]
        x = self.last(x) # [B, S 64, C 3]
        return x


class Encoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.bottleneck = BottleNeck(3, 16, 5, 1, 2)

    def forward(self, x):
        # stage 1
        y = self.bottleneck(x)
        return y

class BottleNeck(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=1, padding=1, expansion=3):
        super().__init__()
        self.h_dim = expansion*in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_dim, self.h_dim, 1),
            nn.BatchNorm2d(self.h_dim),
            nn.SiLU(),
            nn.Conv2d(self.h_dim, self.h_dim, kernel, stride, padding, groups = self.h_dim),
            nn.BatchNorm2d(self.h_dim),
            nn.SiLU(),
            nn.Conv2d(self.h_dim, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim)
        )
        self.projection = nn.Conv2d(self.in_dim, self.out_dim, 1, stride, 0)

    def forward(self, x):
        y = self.conv(x) + self.projection(x)
        return y

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

class LinearDepthBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0):
        super(LinearDepthBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, in_dim, kernel_size, stride, padding, groups = in_dim)
        self.conv2 = nn.Conv1d(in_dim, out_dim, 1)
        self.norm1 = nn.BatchNorm1d(in_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x) 
        x = self.conv2(x)
        return x
