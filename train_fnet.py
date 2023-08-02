import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import time
import math
import os
import json
import fnet as fnet_model
from pathlib import Path
import shutil

os.chdir(Path(__file__).parent)

file_folder = Path(__file__).parent

with open(file_folder/f'config.json',encoding='utf-8') as f:
    config = json.load(f)

fnet_cfg = config['fnet']

dtype = torch.float
device_data = torch.device('cpu')
device_train = torch.device('cuda:0')
device_test = torch.device('cpu')

Material = 'TF'

if Material == 'TF':
    TrainingDataSize = fnet_cfg['TrainingDataSize']
    TestingDataSize = fnet_cfg['TestingDataSize']
    IsParallel = fnet_cfg['IsParallel']
    EpochNum = fnet_cfg['EpochNum']
    TestInterval = fnet_cfg['TestInterval']
    BatchSize = fnet_cfg['BatchSize']
    lr = fnet_cfg['lr']
    if IsParallel:
        BatchSize = BatchSize * torch.cuda.device_count()
        lr = lr * torch.cuda.device_count()
    lr_decay_step = fnet_cfg['lr_decay_step']
    lr_decay_gamma = fnet_cfg['lr_decay_gamma']

    folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = Path('nets/fnet/' + folder_name + '/')
    path.mkdir(parents=True)

    train_data = scio.loadmat(fnet_cfg['TrainDataPath'])
    test_data = scio.loadmat(fnet_cfg['TestDataPath'])
    n_array = train_data['n']
    scio.savemat(path / 'n.mat', {'n': n_array})
    InputNum = train_data['d'].shape[1]
    StartWL = fnet_cfg['StartWL']
    EndWL = fnet_cfg['EndWL']
    Resolution = fnet_cfg['Resolution']
    WL = np.arange(StartWL, EndWL, Resolution)
    OutputNum = WL.size
    assert WL.size == np.array(train_data['wl']).size
    Input_train = torch.tensor(train_data['d'][0:TrainingDataSize], device=device_data, dtype=dtype)
    Output_train = torch.tensor(train_data['T'][0:TrainingDataSize], device=device_data, dtype=dtype)
    Input_test = torch.tensor(test_data['d'][0:TestingDataSize], device=device_test, dtype=dtype)
    Output_test = torch.tensor(test_data['T'][0:TestingDataSize], device=device_test, dtype=dtype)

    del train_data, test_data

fnet = fnet_model.get_fnet(InputNum, OutputNum, model=fnet_cfg.get('model','original'))

if IsParallel:
    fnet = nn.DataParallel(fnet)
fnet.to(device_train)
fnet.train()

LossFcn = nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(fnet.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

loss = torch.tensor([0], device=device_train)
loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))

os.makedirs(path, exist_ok=True)
with open(path/'config.json','w', encoding='utf-8') as cf:
    json.dump(config, cf, indent=4, ensure_ascii=False)
log_file = open(path / 'TrainingLog.txt', 'w+')
time_start = time.time()
time_epoch0 = time_start
for epoch in range(EpochNum):
    idx = torch.randperm(TrainingDataSize, device=device_data)
    Input_train = Input_train[idx, :]
    Output_train = Output_train[idx, :]
    for i in range(0, TrainingDataSize // BatchSize):
        InputBatch = Input_train[i * BatchSize: i * BatchSize + BatchSize, :]
        OutputBatch = Output_train[i * BatchSize: i * BatchSize + BatchSize, :]
        Output_pred = fnet(InputBatch.to(device_train))
        loss = LossFcn(OutputBatch.to(device_train), Output_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    if epoch % TestInterval == 0:
        fnet.to(device_test)
        fnet.eval()
        Out_test_pred = fnet(Input_test)
        fnet.to(device_train)
        fnet.train()
        loss_train[epoch // TestInterval] = loss.data
        loss_t = LossFcn(Output_test, Out_test_pred)
        loss_test[epoch // TestInterval] = loss_t.data
        if epoch == 0:
            time_epoch0 = time.time()
            time_remain = (time_epoch0 - time_start) * EpochNum
        else:
            time_remain = (time.time() - time_epoch0) / epoch * (EpochNum - epoch)
        print('Epoch:', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler.get_lr()[0], '| remaining time: %.0fs (to %s)'
              % (time_remain, time.strftime('%H:%M:%S', time.localtime(time.time() + time_remain))))
        print('Epoch:', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler.get_lr()[0], file=log_file)
time_end = time.time()
time_total = time_end - time_start
m, s = divmod(time_total, 60)
h, m = divmod(m, 60)
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s))
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s), file=log_file)

fnet.eval()
torch.save(fnet, path / 'fnet.pkl')

fnet.to(device_test)
Output_temp = fnet(Input_train[0, :].to(device_test).unsqueeze(0)).squeeze(0)
FigureTrainLoss = LossFcn(Output_train[0, :].to(device_test), Output_temp)
plt.figure()
plt.plot(WL.T, Output_train[0, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='lower right')
plt.savefig(path / 'train')
plt.show()

Output_temp = fnet(Input_test[0, :].to(device_test).unsqueeze(0)).squeeze(0)
FigureTestLoss = LossFcn(Output_test[0, :].to(device_test), Output_temp)
plt.figure()
plt.plot(WL.T, Output_test[0, :].cpu().numpy())
plt.plot(WL.T, Output_temp.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='lower right')
plt.savefig(path / 'test')
plt.show()

print('Training finished!',
      '| loss in figure \'train.png\': %.5f' % FigureTrainLoss.data.item(),
      '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item())
print('Training finished!',
      '| loss in figure \'train.png\': %.5f' % FigureTrainLoss.data.item(),
      '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item(), file=log_file)
log_file.close()

plt.figure()
plt.plot(range(0, EpochNum, TestInterval), loss_train.detach().cpu().numpy())
plt.plot(range(0, EpochNum, TestInterval), loss_test.detach().cpu().numpy())
plt.semilogy()
plt.legend(['Loss_train', 'Loss_test'], loc='upper right')
plt.savefig(path / 'loss')
plt.show()
