import json
import shutil
from typing import Any
import yaml
import numpy as np
from pathlib import Path

class Fnet_config:
    def __init__(self,config_path):
        with open(config_path,encoding='utf-8') as f:
            fnet_config = json.load(f)['fnet']
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