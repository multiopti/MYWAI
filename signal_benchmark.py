# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:50:25 2022

@author: GustavoSanchez
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

np.random.seed(0)

def make_moons_ndim(
    ndim, 
    radius, 
    number_of_datapoints,
):
    
    X = np.zeros([number_of_datapoints,ndim])
    S = np.zeros(number_of_datapoints*ndim);
    Y = np.zeros(number_of_datapoints)
    for i in range(number_of_datapoints):
        q = np.random.uniform(-1,1,size=ndim)
        q = radius*(1/np.linalg.norm(q))*q
        S[i*ndim:(i+1)*ndim] = q
        if q[0]*q[-1] >= 0:
           y = 0
        else:
           y = 1

        X[i,:] = q
        Y[i] = y

    return X, Y, S


X, Y, S = make_moons_ndim(
    ndim=10,
    radius=1, 
    number_of_datapoints=10000)


# plt.scatter(X[:,0], X[:,1], c=Y)
fig, ax = plt.subplots()
ax.scatter(range(32),S[0:32])
ax.plot(range(32),S[0:32])
plt.show()

# fig1 = plt.figure()
# ax = fig1.add_subplot(projection='3d')
# ax.scatter(X[:,0], X[:,1],X[:,2], c=Y)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y,
  test_size=0.3)

clf = SVC(gamma=1, C=1)

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)


