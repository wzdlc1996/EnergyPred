# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import numpy as np
from ase import Atoms

char = [12, 1, 14, 13]
ele = ["C","H","O","N"]

class enerRegNet(torch.nn.Module):
    def __init__(self, featNum):
        super(enerRegNet, self).__init__()
        self.netEvol = torch.nn.Sequential(
                torch.nn.Linear(featNum,1000),
                torch.nn.BatchNorm1d(1000),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(1000,500),
                torch.nn.BatchNorm1d(500),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(500,1)
        )
        
    def forward(self, x):
        """
        We implement relu functional on the hidden layer
        """
        return torch.sum(self.netEvol(x), 1)
    
class locLoss(torch.nn.Module):
    def __init__(self, atomNum):
        super().__init__()
        self.atomNum = atomNum
    def forward(self, x, y):
        lis = x.reshape(len(y), self.atomNum)
        realv = torch.sum(lis, 1)
        return torch.mean(torch.pow( (realv - y ), 2))

class dataSet(Dataset):
    def __init__(self, valiShare = 0.2):
        mols = ["asp", "eth", "mal", "nap", "sal", "tol", "ura"]
        ele = ["C","H","O","N"]
        self.lenTrain = 0
        self.dataRaw = []
        self.labRaw = []
        self.testRaw = []
        self.dataMols = []
        self.testMols = []
        for molName in mols:
            xall = np.loadtxt("./data/origin/train/"+molName+"_data/coord.dat")
            yall = np.loadtxt("./data/origin/train/"+molName+"_data/ener.dat")
            
            indLis = list(range(len(xall)))
            random.shuffle(indLis)
            self.lenTrain += int((1-valiShare) * len(xall))
            self.dataRaw = xall[indLis]
            self.labRaw = yall[indLis]
            
            
            self.testRaw = np.loadtxt("./data/origin/test/"+molName+"_data/coord.dat")
            self.typeLis = np.int_(np.loadtxt("./data/origin/train/"+molName+"_data/type.dat"))
            self.atomNum = len(self.typeLis)
            self.mol = "".join([ele[i]+str(np.sum(self.typeLis == i)) for i in range(4) if np.sum(self.typeLis == i) != 0])
            self.dataMols = [Atoms(self.mol, np.split(x, len(self.typeLis))) for x in self.dataRaw]
            self.testMols = [Atoms(self.mol, np.split(x, len(self.typeLis))) for x in self.testRaw]
    
    def applyDescriptor(self, descriptor):
        self.featureNumber = descriptor.get_number_of_features()
        dsData = descriptor.create(self.dataMols)
        
        self.dataTrain = dsData[:self.lenTrain * self.atomNum]
        self.labTrain = self.labRaw[:self.lenTrain]
        self.dataVal = dsData[self.lenTrain * self.atomNum:]
        self.labVal = self.labRaw[self.lenTrain:]
        
        self.testData = descriptor.create(self.testMols)
        
    def __len__(self):
        return len(self.labTrain)
    
    def __getitem__(self, idx):
        return self.dataTrain[idx * self.atomNum : (idx + 1) * self.atomNum], self.labTrain[idx]
        