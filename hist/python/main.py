# -*- coding: utf-8 -*-

import net
import torch
import numpy as np
import copy

x = np.loadtxt("../data_mat/dataTrain.dat")
y = np.loadtxt("../data_mat/valuTrain.dat")
x_val = np.loadtxt("../data_mat/dataVali.dat")
y_val = np.loadtxt("../data_mat/valuVali.dat")

x = torch.as_tensor(x, dtype=torch.float32)
y = torch.as_tensor(y, dtype=torch.float32).reshape(len(y),1)
x_val = torch.as_tensor(x_val, dtype=torch.float32)
y_val = torch.as_tensor(y_val, dtype=torch.float32).reshape(len(y_val), 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x = x.to(device)
y = y.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)

model = net.energyRegNet()
model.to(device)

lossFunc = torch.nn.MSELoss().to(device)
trainLoss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
learR = 1e-5
lossReg = 1.
errlis = np.zeros(100)
model.train()
for t in range(500000):
    ypred = model(x)
    loss = trainLoss(ypred, y)
    print(t, loss.item())
    if t > 100 and t % 100 == 0 and np.average(errlis[-50:]) > np.average(errlis[:50]):
        learR /= 2
        optimizer = torch.optim.Adam(model.parameters(), lr=learR)
        print("update learning rate, now is: "+str(learR))
    errlis[t % 100] = loss.item()
    if(t % 25 == 0):
        model.eval()
        tmpy = model.forward(x_val)
        losstmp = np.sqrt(lossFunc(tmpy, y_val).item())
        if losstmp < lossReg:
            modelReg = copy.deepcopy(model)
            lossReg = losstmp
            torch.save(modelReg, "./model.pt")
        print("the "+str(t)+"-th training result is: "+str(losstmp))
        model.train()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
torch.save(modelReg, "./model_mat.pt")

modelReg.eval()

test = np.loadtxt("../data_mat/dataTest.dat")
test = torch.as_tensor(test, dtype = torch.float32).to(device)

testval = modelReg(test)

with open("./resTest.dat", "w") as f:
    f.write("Id,Predicted\n")
    ind = 1
    for tv in testval:
        f.write(str(ind) + ",\t" + str(tv.item())+"\n")
        ind += 1