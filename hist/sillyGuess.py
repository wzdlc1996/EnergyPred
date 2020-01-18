# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
import random

char = [12, 1, 14, 13]

def disMat(x):
    atNum = len(typeLis)
    cord = np.split(x, atNum)
    disMat = np.zeros(int(atNum * (atNum-1)/2))
    ind = 0
    for i in range(atNum):
        for j in range(i+1, atNum):
            disMat[ind] = np.linalg.norm(cord[i] - cord[j])
            ind += 1
    return np.array(sorted(disMat))

class pcaMat():
    def __init__(self, typ):
        self.typeLis = typ
    def __call__(self, x):
        return disMat(x)
        typeLis = self.typeLis
        atNum = len(typeLis)
        charCord = []
        ind = 0
        for atm in typeLis:
            charCord = np.append(charCord, np.tile(x[ind:ind+3], char[atm]))
            ind += 3
        charCord = np.split(charCord, int(len(charCord)/3))
        mean = np.average(charCord)
        charhd = charCord - mean
        pmat = np.dot(charhd.T, charhd)
        w, v = np.linalg.eigh(pmat)
        cord = np.split(x, atNum)
        hd = cord - mean
        newCord = np.dot(hd, v)
        finData = []
        for i in range(4):
            specis = newCord[typeLis == i]
            nm = [np.linalg.norm(x) for x in specis]
            finData = np.append(finData, specis[np.argsort(nm)])
        return finData.flatten()

def silGuess(xall, yall, typeLis, comSel, xtest):
    
    pool = mp.Pool(4)
    wrpx = pool.map(pcaMat(typeLis), xall)
    wrpt = pool.map(pcaMat(typeLis), xtest)
    pool.close()
    
    res = []
    for t in wrpt:
        diff = wrpx - t
        dis = np.linalg.norm(diff, axis = 1)
        resGuess = np.average(yall[np.argsort(dis)[:comSel]])
        res.append(resGuess)
        
    return np.array(res)


xall = np.loadtxt("./data/origin/train/asp_data/coord.dat")
yall = np.loadtxt("./data/origin/train/asp_data/ener.dat")
typeLis = np.int_(np.loadtxt("./data/origin/train/asp_data/type.dat"))
ind = list(range(len(xall)))
random.shuffle(ind)

xtrain = xall[ind[:9000]]
ytrain = yall[ind[:9000]]
xtest = xall[ind[9001:]]
ytest = yall[ind[9001:]]

z = silGuess(xtrain, ytrain, typeLis, 1, xtest)
print(np.sqrt(np.average( (z - ytest)**2 )))
    