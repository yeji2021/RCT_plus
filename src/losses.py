import torch
import torch.nn.functional as F

#frequency loss w. 1/4 grid
def loss_qf(targets, inputs, o1, o2, R, \
            scale_cx = 4e-2, scale_f=4e-2, scale_e = 2e-1, n_grid = 2):
    # The hyper parameter λ is fixed to 0.04 (from RCT paper)
    loss_cy = color_loss(targets, o1) # Y - Ytilde
    loss_cx = color_loss(inputs, o2) # X - Xtilde
    loss_f = GridFrequencyLoss(targets, o1, n_grid)
    loss_e = entropy_loss(R)
    loss = loss_cy + scale_cx *loss_cx + scale_f * loss_f + 1/(loss_e*scale_e)
    return loss, loss_e

def loss_default(targets, outputs, targets_features, outputs_features, scale_p = 4e-2):
    loss_c = color_loss(targets, outputs)
    loss_p = perceptual_loss(targets_features, outputs_features)
    loss = loss_c + scale_p * loss_p
    return loss

def color_loss(targets, outputs):
    loss = F.l1_loss(targets, outputs)
    return loss

def perceptual_loss(targets, outputs):
    loss = 0.0
    for target, output in zip(targets, outputs):
        loss += F.l1_loss(target, output)
    return loss

def frequency_loss(targets, outputs):
    _, _, h,w = targets.shape
    f_targets = torch.fft.fft2(targets) # 32, 3, 256, 256
    f_outputs = torch.fft.fft2(outputs)
    loss = F.l1_loss(f_targets, f_outputs)/(h*w)
    return loss 

def GridFrequencyLoss(targets, outputs, n_grid):
    # n_grid : number of block of one side(height/width) if 2 ? total grid : 4
    _,_,h,w = targets.shape
    g_h = int(h/n_grid)
    g_w = int(h/n_grid)

    f_loss = 0
    for i in range(n_grid):
        for j in range(n_grid):
            t = targets[:,:, i*g_h : i*g_h + g_h, j*g_w : j*g_w + g_w]
            o = outputs[:,:, i*g_h : i*g_h + g_h, j*g_w : j*g_w + g_w]
            f_t = torch.fft.fft2(t)
            o_t = torch.fft.fft2(o)
            f_loss += F.l1_loss(f_t, o_t)/(g_h*g_w)

    return f_loss/(n_grid*n_grid)

def entropy_loss(R):
    #cosine similarity
    R_t = R.permute(0,2,1) # B X N X C
    (b,_,n) = R.shape

    dot = torch.bmm(R_t, R) # B X N X N
    mag = torch.sum(torch.square(R),dim =1).unsqueeze(2) # B X N x 1
    sim = dot/mag # B X N X N

    eye = torch.eye(n, device = sim.device, requires_grad=True)
    sim = sim*(1-eye)

    softmax = torch.nn.Softmax(dim=0).to(sim.device)
    p = softmax(sim)

    b_entropy = torch.sum(torch.mul(-p, torch.log(p)).reshape(b,-1), dim = 1)
    entropy = torch.mean(b_entropy)
    return entropy


