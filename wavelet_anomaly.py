# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 12:32:12 2022

@author: GustavoSanchez
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sk_dsp_comm.sigsys as ss 

import pywt

# Setup
t = np.linspace(0,10,1024)
f1 = 0.5 # Low frequency signal
x1 = np.cos(2*np.pi*f1*t)
f2 =  5
xr = ss.rect(t-6.5,0.5)
x2 = x1 + 0.3*xr*np.cos(2*np.pi*f2*t)

mode = pywt.Modes.smooth


def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    for i in range(5):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    fig = plt.figure()
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))
    
    return rec_a,rec_d


xa,xd = plot_signal_decomp(x2, 'sym5', "Signal plus anomaly")


tv = np.where(np.abs(xd[2]) > 0.05)
fig, ax2 = plt.subplots()
ax2.add_patch(Rectangle((tv[0][0]-1, -1), 100, 2,color="mistyrose"))
ax2.plot(x2)
