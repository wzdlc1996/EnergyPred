# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import xgboost as xgb
from ase import Atoms
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

mols = ["asp", "eth", "mal", "nap", "sal", "tol", "ura"]

molName = mols[0]

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


x, x_val, y, y_val = train_test_split(datax, yall, test_size = 0.2, random_state = 0)

params = {
                "alpha" : [0.01, 1e-3, 1e-4, 1e-5, 1e-6],
                "gamma" : [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
            }

krc = KernelRidge(kernel = "rbf")
kr = GridSearchCV(krc, params)
kr.fit(x,y)
print(np.sqrt(np.average((y - kr.predict(x))**2)))
print(np.sqrt(np.average((y_val - kr.predict(x_val))**2)))
