import HybridNet
from NoiseLayer import *
import torch
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import math
import time
import os
import json
import shutil
from pathlib import Path
from tmm_acc import coh_tmm_normal_spol_spec_d
from test_hybnet import Hybnet_folder
import argparse

os.chdir(Path(__file__).parent)



parser = argparse.ArgumentParser(
    description='Train a hybrid network with existing design parameters. Inherit hybnet config, using new noise config.'
)
parser.add_argument('-f','--folder',type=str,help='folder name containing response curves')
args = parser.parse_args()

hybnet_folder = Hybnet_folder(Path(args.folder))
config = hybnet_folder.PCSED_cfg
config['training_noise_config'] = hybnet_folder.noise_cfg
with open('config.json', 'r') as f:
    f = json.load(f)
    noise_cfg = f['noise']




dtype = torch.float
device_data = torch.device("cpu")
device_train = torch.device("cuda:0")
device_test = torch.device("cpu")

Material = 'TF'
TrainingDataSize = config['TrainingDataSize']
TestingDataSize = config['TestingDataSize']
BatchSize = config['BatchSize']
EpochNum = config['EpochNum']
TestInterval = config['TestInterval']
lr = config['lr']
lr_decay_step = config['lr_decay_step']
lr_decay_gamma = config['lr_decay_gamma']
beta_range = config['beta_range']
TFNum = config['TFNum']

folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
path = Path('nets/hybnet/' + folder_name + '/')
path.mkdir(parents=True, exist_ok=True)

fnet_folder = Path(config['fnet_folder'])

with open(fnet_folder/'config.json',encoding='utf-8') as f:
    fnet_config = json.load(f)['fnet']

with open(path/'config.json', 'w', encoding='utf-8') as f:
    json.dump(
        {'fnet': fnet_config,'PCSED': config, 'noise': noise_cfg}
        , f, ensure_ascii=False, indent=4)
    
shutil.copy(fnet_folder/'n.mat',path/'n.mat')

fnet_path = fnet_folder/'fnet.pkl'

params_min = torch.tensor([fnet_config['params_min']])
params_max = torch.tensor([fnet_config['params_max']])

StartWL = fnet_config['StartWL']
EndWL = fnet_config['EndWL']
Resolution = fnet_config['Resolution']
WL = np.arange(StartWL, EndWL, Resolution)
SpectralSliceNum = WL.size


Specs_train = torch.zeros([TrainingDataSize, SpectralSliceNum], device=device_data, dtype=dtype)
Specs_test = torch.zeros([TestingDataSize, SpectralSliceNum], device=device_test, dtype=dtype)
data = scio.loadmat(config['TrainDataPath'])
Specs_all = np.array(data['data'])
np.random.shuffle(Specs_all)
Specs_train = torch.tensor(Specs_all[0:TrainingDataSize, :])
data = scio.loadmat(config['TestDataPath'])
Specs_all = np.array(data['data'])
np.random.shuffle(Specs_all)
Specs_test = torch.tensor(
    Specs_all[0:TestingDataSize, :])


del Specs_all, data
assert SpectralSliceNum == Specs_train.size(1)

noise_layer = NoiseLayer(SNR=noise_cfg['SNR'], alpha=noise_cfg['alpha'], bitdepth=noise_cfg['bitdepth'])

hybnet_size = [SpectralSliceNum, TFNum, 500, 500, SpectralSliceNum]
hybnet_folder.load_model(device_train)
hybnet_folder.change_noise_layer(SNR=noise_cfg['SNR'], alpha=noise_cfg['alpha'], bitdepth=noise_cfg['bitdepth'])
hybnet:HybridNet.NoisyHybridNet = hybnet_folder.model
hybnet.DesignParams.requires_grad = False
hybnet.SWNet = HybridNet.SWNet(hybnet_size,device_train)





LossFcn = HybridNet.HybnetLoss()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, hybnet.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

loss = torch.tensor([0], device=device_train)
loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))


log_file = open(path / 'TrainingLog.txt', 'w+')
time_start = time.time()
time_epoch0 = time_start
for epoch in range(EpochNum):
    Specs_train = Specs_train[torch.randperm(TrainingDataSize), :]
    for i in range(0, TrainingDataSize // BatchSize):
        Specs_batch = Specs_train[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
        Output_pred = hybnet(Specs_batch)
        DesignParams = hybnet.show_design_params()
        loss = LossFcn(Specs_batch, Output_pred, DesignParams, params_min.to(device_train), params_max.to(device_train), beta_range)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    scheduler.step()
    if epoch % TestInterval == 0:
        hybnet.to(device_test)
        hybnet.eval()
        Out_test_pred = hybnet(Specs_test)
        hybnet.to(device_train)
        hybnet.train()
        hybnet.eval_fnet()
        loss_train[epoch // TestInterval] = loss.data
        loss_t = HybridNet.MatchLossFcn(Specs_test, Out_test_pred)
        loss_test[epoch // TestInterval] = loss_t.data
        if epoch == 0:
            time_epoch0 = time.time()
            time_remain = (time_epoch0 - time_start) * EpochNum
        else:
            time_remain = (time.time() - time_epoch0) / epoch * (EpochNum - epoch)
        print('Epoch: ', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler.get_lr()[0], '| remaining time: %.0fs (to %s)'
              % (time_remain, time.strftime('%H:%M:%S', time.localtime(time.time() + time_remain))))
        print('Epoch: ', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler.get_lr()[0], file=log_file)
time_end = time.time()
time_total = time_end - time_start
m, s = divmod(time_total, 60)
h, m = divmod(m, 60)
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s))
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s), file=log_file)

hybnet.eval()
hybnet.eval_fnet()
torch.save(hybnet, path / 'hybnet.pkl')
hybnet.to(device_test)

HWweights = hybnet.show_hw_weights()
TargetCurves = HWweights.double().detach().cpu().numpy()
scio.savemat(path / 'TargetCurves.mat', mdict={'TargetCurves': TargetCurves})

DesignParams = hybnet.show_design_params()
print(DesignParams[0, :])
TargetCurves_FMN = hybnet.run_fnet(DesignParams).double().detach().cpu().numpy()
scio.savemat(path / 'TargetCurves_FMN.mat', mdict={'TargetCurves_FMN': TargetCurves_FMN})
Params = DesignParams.double().detach().cpu().numpy()
scio.savemat(path / 'TrainedParams.mat', mdict={'Params': Params})

plt.figure()
for i in range(TFNum):
    plt.subplot(math.ceil(math.sqrt(TFNum)), math.ceil(math.sqrt(TFNum)), i + 1)
    plt.plot(WL, TargetCurves[i, :], WL, TargetCurves_FMN[i, :])
    plt.ylim(0, 1)
plt.savefig(path / 'ROFcurves')
plt.show()

Output_train = hybnet(Specs_train[0, :].to(device_test).unsqueeze(0)).squeeze(0)
FigureTrainLoss = HybridNet.MatchLossFcn(Specs_train[0, :].to(device_test), Output_train)
plt.figure()
plt.plot(WL, Specs_train[0, :].cpu().numpy())
plt.plot(WL, Output_train.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='upper right')
plt.savefig(path / 'train')
plt.show()

Output_test = hybnet(Specs_test[0, :].to(device_test).unsqueeze(0)).squeeze(0)
FigureTestLoss = HybridNet.MatchLossFcn(Specs_test[0, :].to(device_test), Output_test)
plt.figure()
plt.plot(WL, Specs_test[0, :].cpu().numpy())
plt.plot(WL, Output_test.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='upper right')
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
