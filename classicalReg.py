# -*- coding: utf-8 -*-

from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import numpy as np
import multiprocessing as mp
import random

mols = ["asp", "eth", "mal", "nap", "sal", "tol", "ura"]

testval = []

for molName in mols:

    xall = np.loadtxt("./data/origin/train/"+molName+"_data/coord.dat")
    yall = np.loadtxt("./data/origin/train/"+molName+"_data/ener.dat")
    typeLis = np.int_(np.loadtxt("./data/origin/train/"+molName+"_data/type.dat"))
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
        return sorted(disMat)
    
    def covMat(x):
        atNum = len(typeLis)
        cord = np.split(x, atNum)
        covMat = []
        for i in range(atNum):
            for j in range(i+1, atNum):
                for k in range(j+1, atNum):
                    rij = cord[i] - cord[j]
                    rij /= np.linalg.norm(rij)
                    rik = cord[i] - cord[k]
                    rik /= np.linalg.norm(rik)
                    rjk = cord[j] - cord[k]
                    rjk /= np.linalg.norm(rjk)
                    covMat.append(np.dot(rij, rik))
                    covMat.append(np.dot(rij, rjk))
        return np.array(covMat)
    
    def refMat(x):
        atNum = len(typeLis)
        cord = np.split(x, atNum)
        disMat = []
        for i in range(atNum):
            for j in range(i+1, atNum):
                disMat = np.append(disMat, cord[i] - cord[j])
        return np.array(disMat)
    
    def pcaMat(x):
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
    
    def dataWrapper(x):
        #return np.append(disMat(x), covMat(x))
        #return refMat(x)
        #return pcaMat(x)
        return disMat(x)
        
        
    testall = np.loadtxt("./data/origin/test/"+molName+"_data/coord.dat")
    pool = mp.Pool(mp.cpu_count())
    wrpx = pool.map(dataWrapper, xall)
    datatest = pool.map(dataWrapper, testall)
    pool.close()
    
    
    indLis =list(range(10000))
    random.shuffle(indLis)
    train = indLis[:9000]
    test = indLis[9001:]
    wrpx = np.array(wrpx)
    
    x = wrpx[train]
    y = yall[train]
    xVal = wrpx[test]
    yVal = yall[test]
    
    params = {
                "alpha" : [0.01, 1e-3, 1e-4, 1e-5, 1e-6],
                "gamma" : [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
            }
    
    
    krc = KernelRidge(kernel = "rbf")
    kr = GridSearchCV(krc, params)
    kr.fit(x,y)
    print(np.sqrt(np.average((y - kr.predict(x))**2)))
    print(np.sqrt(np.average((yVal - kr.predict(xVal))**2)))
    
    
    
    res = kr.predict(datatest)
    testval = np.append(testval, res)
    
    
with open("./resTest.dat", "w") as f:
    f.write("Id,Predicted\n")
    ind = 1
    for tv in testval:
        f.write(str(ind) + ",\t" + str(tv.item())+"\n")
        ind += 1
