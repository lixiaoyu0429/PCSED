import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

from arch.fnet import DCNN, Reshape, DCNN_3
from arch.ADMM_net import ADMM_net_2D, ADMM_net_2
from arch.GAP_net import GAP_net_2
from load_config import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float

import math
import tqdm

net = GAP_net_2(121, stages=5)

EpochNum = 501

LossFcn = nn.MSELoss()

optimizer_net = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
scheduler_net = torch.optim.lr_scheduler.StepLR(optimizer_net, step_size=lr_decay_step, gamma=lr_decay_gamma) 

loss = torch.tensor([0], device=device_train)
loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))
net.to(device_train)

phi = torch.rand((16,121)).to(device_train)

# Train HybNet
for epoch in tqdm.tqdm(range(EpochNum)):
    # Shuffle training data
    Specs_train = Specs_train[torch.randperm(TrainingDataSize), :]
    for i in range(0, TrainingDataSize // BatchSize):
        # Get batch of training data
        Specs_batch = Specs_train[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
        # sampled = torch.matmul(Specs_batch, phi.T).to(device_train)
        sampled = Specs_batch.unsqueeze(-1)
        # Forward pass through HybNet
        Output_pred = net(sampled, torch.rand(16,121).to(device_train)).sum(dim=2)
        # Calculate loss and backpropagate
        loss = LossFcn(Specs_batch, Output_pred)
        optimizer_net.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_net.step()
    scheduler_net.step()
    if epoch % TestInterval == 0:
        loss_train[epoch // TestInterval] = loss.data
        net.eval()
        # Test HybNet
        for i in range(0, TestingDataSize // BatchSize):
            # Get batch of testing data
            Specs_batch = Specs_test[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
            # sampled = torch.matmul(Specs_batch, phi.T).to(device_train)
            sampled = Specs_batch.unsqueeze(-1)
            # Forward pass through HybNet
            Output_pred = net(sampled, phi).sum(dim=2)
            # Calculate loss
            loss = LossFcn(Specs_batch, Output_pred)
        loss_test[epoch // TestInterval] = loss.data
        net.train()

import matplotlib.pyplot as plt
loss_train = loss_train.cpu().numpy()
loss_test = loss_test.cpu().numpy()
plt.plot(loss_train, label='train')
plt.plot(loss_test, label='test')
plt.legend()
plt.savefig(path / 'loss.png')
plt.close()

# Save trained HybNet
torch.save(net, path / 'admm_net.pkl')
