import torch.nn as nn


def get_fnet(InputNum:int, OutputNum:int)->nn.Sequential:
    fnet = nn.Sequential(
        nn.Linear(InputNum, 200),
        nn.BatchNorm1d(200),
        nn.LeakyReLU(inplace=True),
        nn.Linear(200, 800),
        nn.BatchNorm1d(800),
        nn.LeakyReLU(inplace=True),
        nn.Linear(800, 800),
        nn.Dropout(0.1),
        nn.BatchNorm1d(800),
        nn.LeakyReLU(inplace=True),
        nn.Linear(800, 800),
        nn.Dropout(0.1),
        nn.BatchNorm1d(800),
        nn.LeakyReLU(inplace=True),
        nn.Linear(800, 800),
        nn.Dropout(0.1),
        nn.BatchNorm1d(800),
        nn.LeakyReLU(inplace=True),
        nn.Linear(800, OutputNum),
        nn.Dropout(0.1),
        nn.Sigmoid()
    )
    return fnet