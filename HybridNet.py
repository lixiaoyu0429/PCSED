from typing import Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.modules.module import Module

class SWNet(nn.Sequential):
    def __init__(self,size, device):
        super(SWNet, self).__init__()
        self.add_module('LReLU0', nn.LeakyReLU(inplace=True))
        self.add_module('LReLU0', nn.LeakyReLU(inplace=True))
        for i in range(1, len(size) - 1):
            self.add_module('Linear' + str(i), nn.Linear(size[i], size[i + 1]))
            # self.SWNet.add_module('BatchNorm' + str(i), nn.BatchNorm1d(size[i+1]))
            # self.SWNet.add_module('DropOut' + str(i), nn.Dropout(p=0.2))
            self.add_module('LReLU' + str(i), nn.LeakyReLU(inplace=True))

        self.to(device)

class HybridNet(nn.Module):
    def __init__(self, fnet_path, thick_min, thick_max, size,device):
        super(HybridNet, self).__init__()
        self.fnet = torch.load(fnet_path)
        self.fnet.to(device)
        self.fnet.eval()
        for p in self.fnet.parameters():
            p.requires_grad = False
        self.tf_layer_num = self.fnet.state_dict()['0.weight'].data.size(1)
        self.DesignParams = nn.Parameter(
            (thick_max - thick_min) * torch.rand([size[1], self.tf_layer_num]) + thick_min, requires_grad=True)


        self.SWNet = SWNet(size,device)
        self.to(device)

    def forward(self, data_input):
        sampled = func.linear(data_input, self.fnet(self.DesignParams), None)
        return self.SWNet(sampled)

    def show_design_params(self):
        return self.DesignParams

    def show_hw_weights(self):
        return self.fnet(self.DesignParams)

    def eval_fnet(self):
        self.fnet.eval()
        return 0

    def run_fnet(self, design_params_input):
        return self.fnet(design_params_input)

    def run_swnet(self, data_input, hw_weights_input):
        assert hw_weights_input.size(0) == self.DesignParams.size(0)
        return self.SWNet(func.linear(data_input, hw_weights_input, None))


class NoisyHybridNet(HybridNet):
    def __init__(self, fnet_path, thick_min, thick_max, size, noise_layer,device):
        super(NoisyHybridNet, self).__init__(fnet_path, thick_min, thick_max, size,device)
        
        self.noise_layer = noise_layer
        self.noise_layer.to(device)


    def forward(self, data_input):
        sampled = func.linear(data_input, self.fnet(self.DesignParams), None)
        sampled = self.noise_layer(sampled)
        return self.SWNet(sampled)
    
    def run_swnet(self, data_input, hw_weights_input):
        sampled = self.noise_layer(func.linear(data_input, hw_weights_input, None))
        return self.SWNet(sampled)

class ND_HybridNet(NoisyHybridNet):
    def __init__(self, diff_row ,fnet_path, thick_min, thick_max, size, noise_layer,device):
        super(ND_HybridNet, self).__init__(fnet_path, thick_min, thick_max, size, noise_layer,device)
        self.diff_row = diff_row
        self.original_idx = torch.arange(size[1], device=device)
        self.diff_idx = torch.arange(size[1], device=device).reshape(self.diff_row, -1).roll(1, dims=1).reshape(-1)
        self.to(device)

    def forward(self, data_input):
        response = self.fnet(self.DesignParams)
        sampled = func.linear(data_input, response, None)
        sampled = self.noise_layer(sampled)
        diffed_sampled = sampled[:, self.diff_idx] - sampled[:, self.original_idx]
        return self.SWNet(diffed_sampled)
    
    def run_swnet(self, data_input, hw_weights_input):
        diffed_response = hw_weights_input[:, self.diff_idx] - hw_weights_input[:, self.original_idx]
        sampled = self.noise_layer(func.linear(data_input, diffed_response, None))
        return self.SWNet(sampled)



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
