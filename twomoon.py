# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 13:32:11 2022

@author: GustavoSanchez
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

n =10 #double of size of the dataset
np.random.seed(0)

# Generate n points between -pi/2 and pi/2
Theta1 = np.random.rand(n)*np.pi - np.pi/2

# Generate n points between pi/2 and 3pi/2
Theta2 = np.random.rand(n)*np.pi + np.pi/2

R = 80
C1 = [90, 90]
C2 = [150, 150]
noise = 0

X1 = np.zeros((n, 2))
X1[:, 0] = R*np.cos(Theta1) + C1[0] + noise*np.random.rand(n)
X1[:, 1] = R*np.sin(Theta1) + C1[1] + noise*np.random.rand(n)
X2 = np.zeros((n, 2))
X2[:, 0] = R*np.cos(Theta2) + C2[0] + noise*np.random.rand(n)
X2[:, 1] = R*np.sin(Theta2) + C2[1] + noise*np.random.rand(n)

Y = np.zeros(2*n)
Y[:n] = 0
Y[n+1:] = 1
Y = Y.astype(int)

X = np.concatenate((X1, X2))
X = X.astype(int)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.axis([0, 255, 0, 255])
plt.show()

K = 10
accuracy = 0
kfold = KFold(K,random_state = 0, shuffle=True)
for train, test in kfold.split(X):
	clf = MLPClassifier(random_state=0, max_iter=1000, verbose=False).fit(X[train], Y[train])
	accuracy = accuracy + clf.score(X[test],Y[test])/K
       
       
print("Average accuracy cross-validation = ",accuracy)

