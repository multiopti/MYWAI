# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:29:17 2020

@author: Admin
"""
import numpy as np


def dioph_sys(a,b,c):
    # a = polynomial order 3, decreasing
    # b = polynomial order 2, decreasing
    # c = polynomial order 5, decreasing
    a3 = a[0]
    a2 = a[1]
    a1 = a[2]
    a0 = a[3]
    b2 = b[0]
    b1 = b[1]
    b0 = b[2]
    
    row1 = [a3,0,0,0,0,0]
    row2 = [a2,a3,0,b2,0,0]
    row3 = [a1,a2,a3,b1,b2,0]
    row4 = [a0,a1,a2,b0,b1,b2]
    row5 = [0,a0,a1,0,b0,b1]
    row6 = [0,0,a0,0,0,b0]
    
    S = np.vstack([row1,row2,row3,row4,row5,row6])
    coeff = np.linalg.solve(S, c)
    return coeff

d = 3
A  = [1, -0.9, 0, 0]
B  = [0, 0, 1]
C = [1, 0, 0, 0, 0, 0]
#solve AX + BY = C
coeff = dioph_sys(A, B, C)

X = np.array(coeff[0:d])
mv = np.sum(X*X)
print("Theoretical minimum variance = ", mv)

