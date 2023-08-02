import torch.nn as nn

def get_fnet(InputNum:int, OutputNum:int, model:str='original')->nn.Sequential:
    if model == 'original':
        return fnet(InputNum, OutputNum)
    elif model == 'DCNN':
        return DCNN(InputNum, OutputNum)
    elif model == 'DCNN_2':
        return DCNN_2(InputNum, OutputNum)
    elif model == 'fnet_1500':
        return fnet_1500(InputNum, OutputNum)
    elif model == 'fnet_7layer':
        return fnet_7layer(InputNum, OutputNum)
    


class fnet(nn.Sequential):
    def __init__(self, InputNum:int, OutputNum:int)->nn.Sequential:
        super(fnet, self).__init__()
        self.add_module('Linear0', nn.Linear(InputNum, 200))
        self.add_module('BatchNorm0', nn.BatchNorm1d(200))
        self.add_module('LReLU0', nn.LeakyReLU(inplace=True))
        self.add_module('Linear1', nn.Linear(200, 800))
        self.add_module('BatchNorm1', nn.BatchNorm1d(800))
        self.add_module('LReLU1', nn.LeakyReLU(inplace=True))
        self.add_module('Linear2', nn.Linear(800, 800))
        self.add_module('DropOut2', nn.Dropout(0.1))
        self.add_module('BatchNorm2', nn.BatchNorm1d(800))
        self.add_module('LReLU2', nn.LeakyReLU(inplace=True))
        self.add_module('Linear3', nn.Linear(800, 800))
        self.add_module('DropOut3', nn.Dropout(0.1))
        self.add_module('BatchNorm3', nn.BatchNorm1d(800))
        self.add_module('LReLU3', nn.LeakyReLU(inplace=True))
        self.add_module('Linear4', nn.Linear(800, 800))
        self.add_module('DropOut4', nn.Dropout(0.1))
        self.add_module('BatchNorm4', nn.BatchNorm1d(800))
        self.add_module('LReLU4', nn.LeakyReLU(inplace=True))
        self.add_module('Linear5', nn.Linear(800, OutputNum))
        self.add_module('DropOut5', nn.Dropout(0.1))
        self.add_module('Sigmoid', nn.Sigmoid())

class fnet_1500(nn.Sequential):
    def __init__(self, InputNum:int, OutputNum:int)->nn.Sequential:
        super(fnet_1500, self).__init__()
        self.add_module('Linear0', nn.Linear(InputNum, 200))
        self.add_module('BatchNorm0', nn.BatchNorm1d(200))
        self.add_module('LReLU0', nn.LeakyReLU(inplace=True))
        self.add_module('Linear1', nn.Linear(200, 800))
        self.add_module('BatchNorm1', nn.BatchNorm1d(800))
        self.add_module('LReLU1', nn.LeakyReLU(inplace=True))
        self.add_module('Linear2', nn.Linear(800, 1200))
        self.add_module('DropOut2', nn.Dropout(0.1))
        self.add_module('BatchNorm2', nn.BatchNorm1d(1200))
        self.add_module('LReLU2', nn.LeakyReLU(inplace=True))
        self.add_module('Linear3', nn.Linear(1200, 1500))
        self.add_module('DropOut3', nn.Dropout(0.1))
        self.add_module('BatchNorm3', nn.BatchNorm1d(1500))
        self.add_module('LReLU3', nn.LeakyReLU(inplace=True))
        self.add_module('Linear4', nn.Linear(1500, 1500))
        self.add_module('DropOut4', nn.Dropout(0.1))
        self.add_module('BatchNorm4', nn.BatchNorm1d(1500))
        self.add_module('LReLU4', nn.LeakyReLU(inplace=True))
        self.add_module('Linear5', nn.Linear(1500, OutputNum))
        self.add_module('DropOut5', nn.Dropout(0.1))
        self.add_module('Sigmoid', nn.Sigmoid())

class fnet_1500_7layer(nn.Sequential):
    def __init__(self, InputNum:int, OutputNum:int)->nn.Sequential:
        super(fnet_1500_7layer, self).__init__()
        self.add_module('Linear0', nn.Linear(InputNum, 200))
        self.add_module('BatchNorm0', nn.BatchNorm1d(200))
        self.add_module('LReLU0', nn.LeakyReLU(inplace=True))
        self.add_module('Linear1', nn.Linear(200, 800))
        self.add_module('BatchNorm1', nn.BatchNorm1d(800))
        self.add_module('LReLU1', nn.LeakyReLU(inplace=True))
        self.add_module('Linear2', nn.Linear(800, 1200))
        self.add_module('DropOut2', nn.Dropout(0.1))
        self.add_module('BatchNorm2', nn.BatchNorm1d(1200))
        self.add_module('LReLU2', nn.LeakyReLU(inplace=True))
        self.add_module('Linear3', nn.Linear(1200, 1500))
        self.add_module('DropOut3', nn.Dropout(0.1))
        self.add_module('BatchNorm3', nn.BatchNorm1d(1500))
        self.add_module('LReLU3', nn.LeakyReLU(inplace=True))
        self.add_module('Linear4', nn.Linear(1500, 1500))
        self.add_module('DropOut4', nn.Dropout(0.1))
        self.add_module('BatchNorm4', nn.BatchNorm1d(1500))
        self.add_module('LReLU4', nn.LeakyReLU(inplace=True))
        self.add_module('Linear4', nn.Linear(1500, 1500))
        self.add_module('DropOut4', nn.Dropout(0.1))
        self.add_module('BatchNorm4', nn.BatchNorm1d(1500))
        self.add_module('LReLU4', nn.LeakyReLU(inplace=True))
        self.add_module('Linear4', nn.Linear(1500, 1500))
        self.add_module('DropOut4', nn.Dropout(0.1))
        self.add_module('BatchNorm4', nn.BatchNorm1d(1500))
        self.add_module('LReLU4', nn.LeakyReLU(inplace=True))
        self.add_module('Linear5', nn.Linear(1500, OutputNum))
        self.add_module('DropOut5', nn.Dropout(0.1))
        self.add_module('Sigmoid', nn.Sigmoid())

class fnet_7layer(nn.Sequential):
    def __init__(self, InputNum:int, OutputNum:int)->nn.Sequential:
        super(fnet_7layer, self).__init__()
        self.add_module('Linear0', nn.Linear(InputNum, 200))
        self.add_module('BatchNorm0', nn.BatchNorm1d(200))
        self.add_module('LReLU0', nn.LeakyReLU(inplace=True))
        self.add_module('Linear1', nn.Linear(200, 800))
        self.add_module('BatchNorm1', nn.BatchNorm1d(800))
        self.add_module('LReLU1', nn.LeakyReLU(inplace=True))
        self.add_module('Linear2', nn.Linear(800, 800))
        self.add_module('DropOut2', nn.Dropout(0.1))
        self.add_module('BatchNorm2', nn.BatchNorm1d(800))
        self.add_module('LReLU2', nn.LeakyReLU(inplace=True))
        self.add_module('Linear3', nn.Linear(800, 800))
        self.add_module('DropOut3', nn.Dropout(0.1))
        self.add_module('BatchNorm3', nn.BatchNorm1d(800))
        self.add_module('LReLU3', nn.LeakyReLU(inplace=True))
        self.add_module('Linear4', nn.Linear(800, 800))
        self.add_module('DropOut4', nn.Dropout(0.1))
        self.add_module('BatchNorm4', nn.BatchNorm1d(800))
        self.add_module('LReLU4', nn.LeakyReLU(inplace=True))
        self.add_module('Linear5', nn.Linear(800, 800))
        self.add_module('DropOut5', nn.Dropout(0.1))
        self.add_module('BatchNorm5', nn.BatchNorm1d(800))
        self.add_module('LReLU5', nn.LeakyReLU(inplace=True))
        self.add_module('Linear6', nn.Linear(800, 800))
        self.add_module('DropOut6', nn.Dropout(0.1))        
        self.add_module('BatchNorm6', nn.BatchNorm1d(800))
        self.add_module('LReLU6', nn.LeakyReLU(inplace=True))
        self.add_module('Linear7', nn.Linear(800, OutputNum))
        self.add_module('DropOut7', nn.Dropout(0.1))
        self.add_module('Sigmoid', nn.Sigmoid())

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)
    
class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return x.transpose(0,1)


class DCNN(nn.Module):
    def __init__(self, InputNum:int, OutputNum:int)->nn.Module:
        super(DCNN, self).__init__()

        self.OutputNum = OutputNum

        self.DNN_1 = nn.Sequential(
            nn.Linear(InputNum, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 800),
            nn.BatchNorm1d(800),
            nn.LeakyReLU(inplace=True),
        )

        self.CNN_1 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=101,padding=50),
            nn.Dropout(0.1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1, 1, kernel_size=75,padding=37),
            nn.Dropout(0.1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1, 1, kernel_size=51,padding=25),
            nn.Dropout(0.1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1, 1, kernel_size=25,padding=12),
            nn.Dropout(0.1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1, 1, kernel_size=15,padding=7),
            nn.Dropout(0.1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1, 1, kernel_size=7,padding=3),
            nn.Dropout(0.1),
            nn.Sigmoid(),
        )

        self.DNN_2 = nn.Sequential(
            nn.Linear(800, OutputNum),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.DNN_1(x)
        x = x.view(-1, 1, 800)
        x = self.CNN_1(x)
        x = x.view(-1, 800)
        x = self.DNN_2(x)

        return x


class DCNN_2(nn.Module):
    def __init__(self, InputNum:int, OutputNum:int)->nn.Module:
        super(DCNN_2, self).__init__()

        self.OutputNum = OutputNum

        self.DNN_1 = nn.Sequential(
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
        )

        self.CNN_1 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=101,padding=50),
            nn.Dropout(0.1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1, 1, kernel_size=75,padding=37),
            nn.Dropout(0.1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1, 1, kernel_size=51,padding=25),
            nn.Dropout(0.1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1, 1, kernel_size=25,padding=12),
            nn.Dropout(0.1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1, 1, kernel_size=15,padding=7),
            nn.Dropout(0.1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(1, 1, kernel_size=7,padding=3),
            nn.Dropout(0.1),
            nn.Sigmoid(),
        )

        self.DNN_2 = nn.Sequential(
            nn.Linear(800, OutputNum),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.DNN_1(x)
        x = x.view(-1, 1, 800)
        x = self.CNN_1(x)
        x = x.view(-1, 800)
        x = self.DNN_2(x)

        return x