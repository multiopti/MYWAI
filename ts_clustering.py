# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 08:02:27 2022

@author: GustavoSanchez
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tslearn.clustering import TimeSeriesKMeans

def make_patterns(
    freq, 
    amplitude,
    mv,
    noise,
    winlen,
    nsample,
    sclass):
    
    X = np.zeros([nsample,winlen])
    S = np.zeros(nsample*winlen)
    Y = np.zeros(nsample)
    q = np.zeros(winlen)
    for k in range(nsample):
        for i in range(winlen):
            tv = k*nsample + i
            S[tv] = mv + amplitude*np.cos(2*np.pi*freq*tv) + noise*np.random.uniform(-1,1)
            q[i] = S[tv]
        X[k,:] = q
        Y[k] = sclass
    return X, Y, S

np.random.seed(5)
winlen = 100

X1, Y1, S1 = make_patterns(
    0.0003, 
    1,
    1,
    0.01,
    winlen,
    100,
    0)

X2, Y2, S2 = make_patterns(
    0.0003, 
    1.5,
    0,
    0.01,
    winlen,
    100,
    1)

S = np.hstack([S1,S2])
t = np.arange(len(S))
X = np.vstack([X1,X2])

km = TimeSeriesKMeans(n_clusters=2,metric="dtw",random_state=20).fit(X)
labels = km.fit_predict(X)
print(km.cluster_centers_.shape)

plt.figure

fig, ax = plt.subplots()
for k in range(len(labels)):
    if labels[k] == 0:
        ax.add_patch(Rectangle((k*winlen, -2), winlen, 5,color="aquamarine"))
    if labels[k] == 1:
        ax.add_patch(Rectangle((k*winlen, -2), winlen, 5,color="salmon"))
    if labels[k] == 2:
       ax.add_patch(Rectangle((k*winlen, -2), winlen, 5,color="bisque"))
    if labels[k] == 3:
       ax.add_patch(Rectangle((k*winlen, -2), winlen, 5,color="skyblue"))
ax.plot(t, S, 'b')