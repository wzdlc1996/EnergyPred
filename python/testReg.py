# -*- coding: utf-8 -*-

import net
import torch
import numpy as np
import copy

xall = np.loadtxt("../data/origin/train/asp_data/coord.dat")
yall = np.loadtxt("../data/origin/train/asp_data/ener.dat")

x = torch.as_tensor(xall[:8000], dtype=torch.float32)
y = torch.as_tensor(yall[:8000], dtype=torch.float32)
y = y.reshape(len(y), 1)
x_val = torch.as_tensor(xall[8001:], dtype=torch.float32)
y_val = torch.as_tensor(yall[8001:], dtype=torch.float32)
y_val = y_val.reshape(len(y_val), 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x = x.to(device)
y = y.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)

model = net.energyRegNet()
model.to(device)

lossFunc = torch.nn.MSELoss().to(device)
trainLoss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
learR = 1e-4
lossReg = 1.
errlis = np.zeros(300)
for t in range(200000):
    ypred = model(x)
    loss = trainLoss(ypred, y)
    print(t, loss.item())
    if t > 300 and t % 300 == 0 and np.average(errlis[-100:]) > np.average(errlis[:200]):
        learR /= 2
        optimizer = torch.optim.Adam(model.parameters(), lr=learR)
        print("update learning rate, now is: "+str(learR))
    errlis[t % 100] = loss.item()
    if(t % 25 == 0):
        tmpy = model.forward(x_val)
        losstmp = np.sqrt(lossFunc(tmpy, y_val).item())
        if losstmp < lossReg:
            modelReg = copy.deepcopy(model)
            lossReg = losstmp
            torch.save(modelReg, "./model.pt")
        print("the "+str(t)+"-th training result is: "+str(losstmp))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
torch.save(modelReg, "./model_1kx500.pt")

test = np.loadtxt("./data_field/dataTest.dat")
test = torch.as_tensor(test, dtype = torch.float32).to(device)

testval = model(test)

with open("./resTest.dat", "w") as f:
    f.write("Id,Predicted\n")
    ind = 1
    for tv in testval:
        f.write(str(ind) + ",\t" + str(tv.item())+"\n")
        ind += 1