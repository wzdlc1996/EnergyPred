# -*- coding: utf-8 -*-

from ase import Atoms
import numpy as np

xall = np.loadtxt("./data/origin/train/asp_data/coord.dat")
yall = np.loadtxt("./data/origin/train/asp_data/ener.dat")
typeLis = np.int_(np.loadtxt("./data/origin/train/asp_data/type.dat"))
char = [12, 1, 14, 13]
ele = ["C","H","O","N"]
mol = "".join([ele[i]+str(np.sum(typeLis == i)) for i in range(4) if np.sum(typeLis == i) != 0])

molecule = Atoms(mol, np.split(xall[0], len(typeLis)))

#from ase.visualize import view
#view(molecule)

from dscribe.descriptors import ACSF

# Setting up the ACSF descriptor
acsf = ACSF(
    species=["C", "H", "O"],
    rcut=6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)