# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


class energyRegNet(torch.nn.Module):
    def __init__(self):
        super(energyRegNet, self).__init__()
        self.netEvol = torch.nn.Sequential(
                torch.nn.Linear(21*11,1000),
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
        return self.netEvol(x)
        