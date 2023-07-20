import torch
import torch.nn as nn
import torch.nn.functional as func

class SWNet(nn.Module):
    def __init__(self, size, device):
        super(SWNet, self).__init__()
        self.SWNet = nn.Sequential()
        self.SWNet.add_module('LReLU0', nn.LeakyReLU(inplace=True))
        for i in range(1, len(size) - 1):
            self.SWNet.add_module('Linear' + str(i), nn.Linear(size[i], size[i + 1]))
            self.SWNet.add_module('LReLU' + str(i), nn.LeakyReLU(inplace=True))
        self.to(device)

    def forward(self, data_input):
        return self.SWNet(data_input)


class PCSED(nn.Module):
    def __init__(self, fnet:nn.Module, rnet:nn.Module, config, device):
        super(PCSED, self).__init__()
        self.fnet = fnet
        self.fnet.to(device)
        self.fnet.eval()
        for p in self.fnet.parameters():
            p.requires_grad = False
        self.tf_layer_num = self.fnet.state_dict()['0.weight'].data.size(1)


        self.rnet = rnet
        self.config = config['PCSED']
        thick_min = self.config['thick_min']
        thick_max = self.config['thick_max']
        size = self.config['size']

        self.DesignParams = nn.Parameter(
            (thick_max - thick_min) * torch.rand([size[1], self.tf_layer_num]) + thick_min, requires_grad=True)
        
    def forward(self, data_input):
        return self.rnet(func.linear(data_input, self.fnet(self.DesignParams), None))
    
    def show_design_params(self):
        return self.DesignParams
    
    def show_hw_weights(self):
        return self.fnet(self.DesignParams)
    
    def eval_fnet(self):
        self.fnet.eval()
        return 0

    def run_fnet(self, design_params_input):
        return self.fnet(design_params_input)       

MatchLossFcn = nn.MSELoss(reduction='mean')

class HybnetLoss(nn.Module):
    def __init__(self):
        super(HybnetLoss, self).__init__()

    def forward(self, t1, t2, params, thick_min, thick_max, beta_range):
        # MSE loss
        match_loss = MatchLossFcn(t1, t2)

        # Structure parameter range regularization.
        # U-shaped function，U([param_min + delta, param_max - delta]) = 0, U(param_min) = U(param_max) = 1。
        delta = 0.01
        res = torch.max((params - thick_min - delta) / (-delta), (params - thick_max + delta) / delta)
        range_loss = torch.mean(torch.max(res, torch.zeros_like(res)))

        return match_loss + beta_range * range_loss