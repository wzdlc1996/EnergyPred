# -*- coding: utf-8 -*-

from ase import Atoms
import numpy as np

mols = ["asp", "eth", "mal", "nap", "sal", "tol", "ura"]

testval = []

for molName in mols:
    xall = np.loadtxt("./data/origin/train/"+molName+"_data/coord.dat")
    yall = np.loadtxt("./data/origin/train/"+molName+"_data/ener.dat")
    typeLis = np.int_(np.loadtxt("./data/origin/train/"+molName+"_data/type.dat"))
    char = [12, 1, 14, 13]
    ele = ["C","H","O","N"]
    mol = "".join([ele[i]+str(np.sum(typeLis == i)) for i in range(4) if np.sum(typeLis == i) != 0])
    
    molecules = [Atoms(mol, np.split(x, len(typeLis))) for x in xall]
    
    #from ase.visualize import view
    #view(molecule)
    
    from dscribe.descriptors import ACSF
    
    # Setting up the ACSF descriptor
    
    acsf = ACSF(
        species=["C", "H", "O", "N"],
        rcut=3.0,
        g2_params=[[1, 1], [1, 2]],
        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
    )
    
    datax = np.split(acsf.create(molecules).flatten(), len(molecules))
    
    x = datax
    y = yall
    
    from sklearn.kernel_ridge import KernelRidge
    
    kr = KernelRidge(alpha = 1e-6, kernel = "rbf", gamma = 0.5)
    kr.fit(x,y)
    
    testall = np.loadtxt("./data/origin/test/"+molName+"_data/coord.dat")
    typeLis = np.int_(np.loadtxt("./data/origin/test/"+molName+"_data/type.dat"))
    char = [12, 1, 14, 13]
    ele = ["C","H","O","N"]
    mol = "".join([ele[i]+str(np.sum(typeLis == i)) for i in range(4) if np.sum(typeLis == i) != 0])
    
    molecules = [Atoms(mol, np.split(x, len(typeLis))) for x in testall]
    
    datatest = np.split(acsf.create(molecules).flatten(), len(molecules))
    
    res = kr.predict(datatest)
    testval = np.append(testval, res)
    
with open("./resTest.dat", "w") as f:
    f.write("Id,Predicted\n")
    ind = 1
    for tv in testval:
        f.write(str(ind) + ",\t" + str(tv.item())+"\n")
        ind += 1