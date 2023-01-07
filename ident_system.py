# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 21:46:30 2023

@author: GustavoSanchez
"""

#%% Import libraries
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal



#%% Generate inputa data
np.random.seed(1234)
u = np.random.uniform(0.2,0.8,40) 
u = np.repeat(u, 100)
t = np.arange(0, 400,0.1)

N = [10]
D = [1, 2 , 10]
sys = sp.signal.lti(N,D)

tout, y, x = signal.lsim(sys, u, t)

u1 = np.roll(u,1)
u1[0] = 0

y1 = np.roll(y,1)
y1[0] = 0

y2 = np.roll(y1,1)
y2[0] = 0

data = [u,u1,y1,y2,y]
data = np.array(data)
data = data.T

# f, (ax1, ax2) = plt.subplots(2, sharey=True)
# ax1.plot(tout,u)
# ax2.plot(tout,y)


Xtrain = data[0:2800,0:4]
ytrain = data[0:2800,4]

Xtest = data[2800:-1,0:4]
ytest = data[2800:-1,4]

#%% Define model

model = MLPRegressor(hidden_layer_sizes=(1,),n_iter_no_change=100, activation='identity', random_state=2,tol=1e-5, max_iter=2000,verbose=True).fit(Xtrain, ytrain)

model.fit(Xtrain, ytrain)

#%%  demonstrate prediction
ypred = model.predict(Xtest)

plt.plot(t[2800:-1],ypred,label='Prediction',color ='r')
plt.plot(t[2800:-1],ytest,label='Measured', color = 'k')
plt.xlabel('Time(s)')
plt.legend()
plt.show()


