# -*- coding: utf-8 -*-

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import numpy as np
"""
xall = np.loadtxt("../data/origin/train/asp_data/coord.dat")
yall = np.loadtxt("../data/origin/train/asp_data/ener.dat")
typeLis = np.int_(np.loadtxt("../data/origin/train/asp_data/type.dat"))
char = [12, 1, 14, 13]

x = xall[:8000]
y = yall[:8000]
x_val = xall[8001:]
y_val = yall[8001:]
"""

x = np.loadtxt("../data_mat/dataTrain.dat")
y = np.loadtxt("../data_mat/valuTrain.dat")
x_val = np.loadtxt("../data_mat/dataVali.dat")
y_val = np.loadtxt("../data_mat/valuVali.dat")
    
    

kr = KernelRidge(kernel="rbf", alpha = 0.1)

kr.fit(x,y)
print(np.sqrt(np.average((kr.predict(x_val) - y_val)**2)))