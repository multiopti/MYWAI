# -*- coding: utf-8 -*-

"""
Simulation of Simple pendulum
"""

# Importing libraries

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D



# Initial and end values
st = 0          # Start time (s)
et = 100       # End time (s)
ts = 0.01        # Time step (s)
g = 9.81        # Acceleration due to gravity (m/s^2)
L = 5           # Length of pendulum (m)
b = 0.05         # Damping factor (kg/s)
m = 1           # Mass of bob (kg)


"""
 theta1 is angular displacement at current time instant
 theta2 is angular velocity at current time instant
 dtheta2_dt is angular acceleration at current time instant
 dtheta1_dt is rate of change of angular displacement at current time instant i.e. same as theta2 
"""

def sim_pen_eq(t,theta):
		dtheta2_dt = (-b/m)*theta[1] + (-g/L)*np.sin(theta[0])
		dtheta1_dt = theta[1]
		return [dtheta1_dt, dtheta2_dt]



# main

theta1_ini = 1                 # Initial angular displacement (rad)
theta2_ini = 0                 # Initial angular velocity (rad/s)
theta_ini = [theta1_ini, theta2_ini]
t_span = [st,et+ts]
t = np.arange(st,et+ts,ts)
sim_points = len(t)
l = np.arange(0,sim_points,1)

theta12 = solve_ivp(sim_pen_eq, t_span, theta_ini, t_eval = t)
theta1 = theta12.y[0,:]
theta2 = theta12.y[1,:]


# # split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

n_steps = 50
n_features = 1
theta1 = theta1 + np.random.normal(0,0.02,size=theta1.shape)
# split into samples
X, y = split_sequence(theta1, n_steps)

# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# fit model
model.fit(X[0:7000,:], y[0:7000], epochs=10, verbose=1)
# demonstrate prediction
x_input = X[7000:-1,:]
yhat = model.predict(x_input, verbose=1)

yd = np.roll(theta1,-50)
plt.plot(t[7000:-51],yd[7000:-51],label='Angular Displacement (rad)',color ='k')
plt.plot(t[7000:-51],yhat,label='Prediction', color = 'r')
plt.xlabel('Time(s)')
plt.legend()
plt.show()
