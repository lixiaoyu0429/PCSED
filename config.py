import json
import shutil
from typing import Any
import yaml
import numpy as np
from pathlib import Path

class Fnet_config:
    def __init__(self,config_path:Path):

        fnet_config = {}
        if config_path.suffix == '.json':
            with open(config_path,encoding='utf-8') as f:
                f = json.load(f)
                fnet_config = f['fnet']
        elif config_path.suffix == '.yml':
            with open(config_path,encoding='utf-8') as f:
                f = yaml.load(f,Loader=yaml.FullLoader)
                fnet_config = f['fnet']


        self.params_min = fnet_config['params_min']
        self.params_max = fnet_config['params_max']
        self.StartWL = fnet_config['StartWL']
        self.EndWL = fnet_config['EndWL']
        self.Resolution = fnet_config['Resolution']
        self.fnet_folder = Path(fnet_config['fnet_folder'])
        self.fnet_path = self.fnet_folder/'fnet.pkl'
        self.n_path = self.fnet_folder/'n.mat'
        self.fnet_config = fnet_config
        self.fnet_config_path = config_path
        self.WL = np.arange(self.StartWL, self.EndWL, self.Resolution)

    def __getitem__(self, key):
        return self.fnet_config[key]
    
class PCSED_config:
    def __init__(self, config_path):
        config = {}

        if config_path.suffix == '.json':
            with open(config_path,encoding='utf-8') as f:
                f = json.load(f)
                config = f['PCSED']
        elif config_path.suffix == '.yml':
            with open(config_path,encoding='utf-8') as f:
                f = yaml.load(f,Loader=yaml.FullLoader)
                config = f['PCSED']


        self.TrainingDataSize = config['TrainingDataSize']
        self.TestingDataSize = config['TestingDataSize']
        self.BatchSize = config['BatchSize']
        self.EpochNum = config['EpochNum']
        self.TestInterval = config['TestInterval']
        self.lr = config['lr']
        self.lr_decay_step = config['lr_decay_step']
        self.lr_decay_gamma = config['lr_decay_gamma']
        self.beta_range = config['beta_range']
        self.TFNum = config['TFNum']
        self.config = config
        self.config_path = config_path

    def __getitem__(self, key):
        return self.config[key]
    
class Noise_config:
    def __init__(self, config_path):
        try:
            noise_cfg = {}
            if config_path.suffix == '.json':
                with open(config_path,encoding='utf-8') as f:
                    f = json.load(f)
                    noise_cfg = f['noise']
            elif config_path.suffix == '.yml':
                with open(config_path,encoding='utf-8') as f:
                    f = yaml.load(f,Loader=yaml.FullLoader)
                    noise_cfg = f['noise']

            self.noise_cfg = noise_cfg
            self.config_path = config_path
        except KeyError:
            self.noise_cfg = {}
            self.config_path = config_path

    def __getitem__(self, key):
        return self.noise_cfg[key]
    
class Config:
    def __init__(self, config_path):
        self.fnet_config = Fnet_config(config_path)
        self.PCSED_config = PCSED_config(config_path)
        self.noise_config = Noise_config(config_path)
        self.config_path = config_path

    