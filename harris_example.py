# -*- coding: utf-8 -*-
"""Control Loop Performance Assessment 
"""

from matplotlib.pyplot import * # Grab MATLAB plotting functions
from control.matlab import *    # MATLAB-like functions
import numpy as np
from scipy import signal
from statsmodels.tsa.arima.model import ARIMA

# Plant model
td = 3 #pure delay

B = 1
a1 = 1
a0 = -0.9
A  = [a1,a0,0,0]

final_p = 10000
sampling_t = 1
t = np.arange(0.0, final_p, sampling_t)
Ft = len(t)

# Closed-loop simulation
Gp = minreal(tf(B, A,sampling_t))
Gc = tf([0.1],[1],sampling_t)
Gycl = minreal(Gp/(1+Gp*Gc))


# epsilon = np.zeros(Ft)
# winlen = 10
# for k in range(int(Ft/winlen)):
#     epsilon[k*winlen:(k+1)*winlen] = np.random.normal(0,1)*np.ones(winlen)
epsilon = np.random.normal(0,1,size = Ft)
ysim, tout, x = lsim(Gycl, epsilon.T, t)
yol, tout, x = lsim(Gp, epsilon.T, t)
ylim((-15, 15))
plot(ysim[0:100],'k')
# plot(yol[0:100])
ydata = signal.detrend(ysim)
order_AR = 4
order_MA = 0
arma_mod = ARIMA(ydata, order=(order_AR, 0, order_MA))
arma_res = arma_mod.fit()
print(arma_res.summary())

AR = arma_mod._polynomial_ar
MA = arma_mod._polynomial_ma
MA = np.hstack([MA,np.zeros(td+order_AR - order_MA)])
Q,R = signal.deconvolve(MA, AR)

mv = np.sum(Q*Q)
yvar = np.var(ydata)
harris = 100*mv/yvar
print("Harris index = ",harris)



