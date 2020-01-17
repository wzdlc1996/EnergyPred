# -*- coding: utf-8 -*-

import mlUtils as utl
import torch
import numpy as np
import copy

from dscribe.descriptors import ACSF
from torch.utils.data import DataLoader


mols = ["asp", "eth", "mal", "nap", "sal", "tol", "ura"]

acsf = ACSF(
    species=["C", "H", "O", "N"],
    rcut=10.0,
    g2_params=[[1, 1], [1, 2]]
)

featNum = acsf.get_number_of_features()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = utl.enerRegNet(featNum)
model.to(device)

datalds = []
atomNums = []

validations = []

for molName in mols:
    
    datas = utl.dataSet(molName)
    datas.applyDescriptor(acsf)
    atomNums.append(datas.atomNum)
    datalds.append(DataLoader(datas, batch_size=32, shuffle=True, num_workers= 4))
    
    x_val = torch.as_tensor(datas.dataVal)
    y_val = torch.as_tensor(datas.labVal)
    
    validations.append((x_val, y_val))

lossFuncs = []
for atomNum in atomNums:
    lossFuncs.append(utl.locLoss(atomNum).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
learR = 1e-3
lossReg = 1.
errlis = np.zeros(100)
model.train()
for t in range(5000):
    cost = 0
    num = 0
    for i in range(len(datalds)):
        for data in datalds[i]:
            btSize = datalds[i].batch_size
            atomNum = atomNums[i]
            x = data[0].reshape(atomNum * btSize, featNum).to(device)
            y = data[1].to(device)
            ypred = model(x)
            loss = lossFuncs[i](ypred, y)
            
            del x
            del y
            
            num += 1
            cost += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    print(t, np.sqrt(cost / num))
        
    if t != 0 and (t % 50 == 0):
        model.eval()
        
        with torch.no_grad():
            losstmps = []
            for i in range(len(validations)):
                x_val = validations[i][0].to(device)
                y_val = validations[i][1].to(device)
                tmpy = model(x_val)
                losstmps.append(lossFuncs[i](tmpy, y_val).item())
                
                del x_val
                del y_val
                
            losstmp = np.sqrt(np.average(losstmps))
            if losstmp < lossReg:
                modelReg = copy.deepcopy(model)
                lossReg = losstmp
                torch.save(modelReg, "./model_local.pt")
                print("saved the model")
            
        print("the "+str(t)+"-th training result is: "+str(losstmp))
        print("the detail: "+str(losstmps))
        model.train()
