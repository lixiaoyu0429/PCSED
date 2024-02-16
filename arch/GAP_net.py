import torch
import torch.nn as nn
import torch.nn.functional as F
from .NoiseLayer import NoiseLayer

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

class MyModel(nn.Module):
    def __init__(self,channel=26):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(channel, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 150)
        self.fc4 = nn.Linear(150, 150)
        self.fc5 = nn.Linear(150, channel)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        x = x.view(x.size(0), x.size(1), 1)
        return x

class GAP_net_2(nn.Module):
    def __init__(self, channel=26,stages=9, noiselayer=NoiseLayer(100,0.1)):
        super(GAP_net_2, self).__init__()
        self.stages = stages

        # self.gamma = torch.nn.Parameter(torch.Tensor([10]*stages))

        for i in range(stages):
            setattr(self, f'net{i}', MyModel(channel))

        self.noiselayer = noiselayer


    def forward(self, imgs, Phi): #, Train=True):
        # imgs: [batch, 121, 1]
        PhiT = torch.transpose(Phi, dim0=0, dim1=1) # [121, 9]
        Phi_PhiT = torch.matmul(Phi, PhiT) # [9, 9]
        invPPT = torch.inverse(Phi_PhiT)
        PT_invPPT = torch.matmul(PhiT, invPPT)

        clear_y = torch.matmul(Phi, imgs) # [batch, 9,1]
        y = self.noiselayer(clear_y) # [batch, 9,1]

        x = torch.matmul(PhiT, y) 
        for i in range(self.stages):
            yb = torch.matmul(Phi, x)
            x = x + torch.matmul(PT_invPPT, y-yb)
            x = getattr(self, f'net{i}')(x)


        return x
    
if __name__=='__main__':
    imgs = torch.rand((10,121,1))
    Phi = torch.rand((9,121))
    net = GAP_net_2(121, stages=5)
    out = net(imgs, Phi)