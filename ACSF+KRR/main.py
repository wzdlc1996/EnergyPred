# -*- coding: utf-8 -*-

from ase import Atoms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from dscribe.descriptors import ACSF
    
mols = ["asp", "eth", "mal", "nap", "sal", "tol", "ura"]

testval = []

for molName in mols:
    xall = np.loadtxt("./data/origin/train/"+molName+"_data/coord.dat")
    yall = np.loadtxt("./data/origin/train/"+molName+"_data/ener.dat")
    typeLis = np.int_(np.loadtxt("./data/origin/train/"+molName+"_data/type.dat"))
    char = [12, 1, 14, 13]
    ele = ["C","H","O","N"]
    mol = "".join([ele[i]+str(np.sum(typeLis == i)) for i in range(4) if np.sum(typeLis == i) != 0])
    
    # Generate Atoms objects by ase
    molecules = [Atoms(mol, np.split(x, len(typeLis))) for x in xall]
    
    # Setting up the ACSF descriptor
    acsf = ACSF(
        species=[ele[i] for i in range(4) if np.sum(typeLis == i) != 0],
        rcut=8.0,
        g2_params=[[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [3, 2]],
        g4_params=[[1, 1, 1], [1, 1, -1]],
    )
    
    # Encode all input
    datax = np.split(acsf.create(molecules).flatten(), len(molecules))
    
    # Generate the trainning set and validation set
    x, x_val, y, y_val = train_test_split(datax, yall, test_size = 0.2, random_state = 0)

    params = {
                    "alpha" : [0.01, 0.1, 1.],
                    "gamma" : [0.1, 0.5, 1.0, 0.05, 0.3, 0.7, 0.01, 0.05, 0.07]
                }
    
    krc = KernelRidge(kernel = "rbf")
    kr = GridSearchCV(krc, params)
    kr.fit(x,y)
    
    # Print the RMSE on training set
    print(np.sqrt(np.average((y - kr.predict(x))**2)))
    
    # Print the RMSE on validation set
    print(np.sqrt(np.average((y_val - kr.predict(x_val))**2)))
    
    # Load and encode the test set
    testall = np.loadtxt("./data/origin/test/"+molName+"_data/coord.dat")
    typeLis = np.int_(np.loadtxt("./data/origin/test/"+molName+"_data/type.dat"))
    mol = "".join([ele[i]+str(np.sum(typeLis == i)) for i in range(4) if np.sum(typeLis == i) != 0])
    
    molecules = [Atoms(mol, np.split(x, len(typeLis))) for x in testall]
    
    datatest = np.split(acsf.create(molecules).flatten(), len(molecules))
    
    res = kr.predict(datatest)
    testval = np.append(testval, res)

# Write the output file in sample.csv format    
with open("./resTest.dat", "w") as f:
    f.write("Id,Predicted\n")
    ind = 1
    for tv in testval:
        f.write(str(ind) + ",\t" + str(tv.item())+"\n")
        ind += 1