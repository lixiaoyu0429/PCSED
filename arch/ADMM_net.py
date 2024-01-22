import torch
import torch.nn as nn
import torch.nn.functional as F
def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)

        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)

        x = self.upsample2(conv3)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs

        return out[:, :, :h_inp, :w_inp]

def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

class ADMM_net(nn.Module):

    def __init__(self, shape=(256,310,28), stages=9):
        super(ADMM_net, self).__init__()

        self.shape = shape
        self.h, self.w, self.c = shape

        self.unet1 = Unet(self.c, self.c)
        self.unet2 = Unet(self.c, self.c)
        self.unet3 = Unet(self.c, self.c)
        self.unet4 = Unet(self.c, self.c)
        self.unet5 = Unet(self.c, self.c)
        self.unet6 = Unet(self.c, self.c)
        self.unet7 = Unet(self.c, self.c)
        self.unet8 = Unet(self.c, self.c)
        self.unet9 = Unet(self.c, self.c)
        # self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        # self.gamma2 = torch.nn.Parameter(torch.Tensor([0]))
        # self.gamma3 = torch.nn.Parameter(torch.Tensor([0]))
        # self.gamma4 = torch.nn.Parameter(torch.Tensor([0]))
        # self.gamma5 = torch.nn.Parameter(torch.Tensor([0]))
        # self.gamma6 = torch.nn.Parameter(torch.Tensor([0]))
        # self.gamma7 = torch.nn.Parameter(torch.Tensor([0]))
        # self.gamma8 = torch.nn.Parameter(torch.Tensor([0]))
        # self.gamma9 = torch.nn.Parameter(torch.Tensor([0]))
        self.gammas = torch.nn.Parameter(torch.Tensor([0]*stages))

    def forward(self, y, input_mask=None):
        if input_mask == None:
            Phi = torch.rand((1, self.shape[2], self.shape[0], self.shape[1])).cuda() # [1, 28, 256, 310]
            Phi_s = torch.rand((1, self.shape[0], self.shape[1])).cuda()
        else:
            Phi, Phi_s = input_mask
        x_list = []
        theta = At(y,Phi) # [1, 28, 256, 310]
        b = torch.zeros_like(Phi)

        for i in range(self.stages):
            yb = A(theta+b,Phi)
            # x = theta+b + At(torch.div(y-yb,Phi_s+getattr(self, f"gamma{i+1}")),Phi)
            x = theta+b + At(torch.div(y-yb,Phi_s+self.gammas[i]),Phi)
            x1 = x-b
            x1 = shift_back_3d(x1)
            theta = getattr(self, f"unet{i+1}")(x1)
            theta = shift_3d(theta)
            b = b- (x-theta)
            x_list.append(theta)

        return theta[:, :, :, 0:self.shape[0]]

class ADMM_net_2D(nn.Module):
    def __init__(self, n_channel=121,stages=9):
        super(ADMM_net_2D, self).__init__()
        self.gammas = torch.nn.Parameter(torch.Tensor([0]*stages))
        self.stages = stages
        self.n_channel = n_channel

    def forward(self, y, Phi):

        def At(y,Phi):
            temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1)
            x = temp*Phi
            return x

        x_list = []
        theta = At(y,Phi)
        b = torch.zeros_like(Phi)

        for i in range(self.stages):
            yb = A(theta+b,Phi)
            x = theta+b + A(torch.div(y-yb,Phi+self.gammas[i]),Phi)
            theta = x-b
            b = b- (x-theta)
            x_list.append(theta)

        return theta