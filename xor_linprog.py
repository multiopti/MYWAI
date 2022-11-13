# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 09:48:35 2022

@author: GustavoSanchez
"""
from scipy.optimize import milp
from scipy.optimize import Bounds
import numpy as np
from scipy.optimize import LinearConstraint
import matplotlib.pyplot as plt

# Solving XOR problem using linear programming

#initial iteration
print("Iteration 1")
eps = 0.1
Big = 100

A = [[0,0],[1,1]]
B = [[0,1],[1,0]]

obj = [-1, 1, 0, 0, 0] # Objective function
lhs_ineq = [[1, 0, 0, 0, 0],
            [1, 0, -1, -1, 0],
            [0, -1, 0, 1, 0,],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 0, -Big],
            [0, 0, 0, -1, -Big],
            [0, 0, 1, 0, Big],
            [0, 0, 0, 1, Big]]

b_u    =   [ 0,
             0,
             0,
             0,
             -eps,
             -eps,
             Big -eps,
             Big -eps]
b_l = np.full_like(b_u, -np.inf)
constraints = LinearConstraint(lhs_ineq, b_l, b_u)

integr = [0,0,0,0,1]

x0_b = (-Big, Big)
x1_b = (-Big, Big)
x2_b = (-1, 1)
x3_b = (-1, 1)
x4_b = (0, 1)

Bnd = Bounds(lb=[-Big,-Big,-1,-1,0], ub=[Big,Big,1,1,1])
res = milp(c=obj, constraints=constraints, integrality=integr,bounds=Bnd)

print(np.dot(A[0],res.x[2:4]))
print(np.dot(B[0],res.x[2:4]))
print(np.dot(B[1],res.x[2:4]))
print(np.dot(A[1],res.x[2:4]))
print("Classified in class A ", A[1] )
print("Parameters = ",res.x)

t = np.arange(-0.5,1.2,0.1)
L2 = -res.x[2]*t/res.x[3] + res.x[1]/res.x[3]

# plt.ylim([0, 1])
# plt.xlim([0, 1.2])
# plt.plot(t,L2)

print("Iteration 2")
#second iteration
A = [[0,0]]
B = [[0,1],[1,0]]

lhs_ineq = [[1, 0, 0, 0, 0],
            [0, -1, 0, 1, 0,],
            [0, -1, 1, 0, 0],
            [0, 0, -1, 0, -Big],
            [0, 0, 0, -1, -Big],
            [0, 0, 1, 0, Big],
            [0, 0, 0, 1, Big]]

b_u    =   [ 0,
             0,
             0,
             -eps,
             -eps,
             Big -eps,
             Big -eps]
b_l = np.full_like(b_u, -np.inf)
constraints = LinearConstraint(lhs_ineq, b_l, b_u)

res = milp(c=obj, constraints=constraints, integrality=integr,bounds=Bnd)

print(np.dot(A[0],res.x[2:4]))
print(np.dot(B[0],res.x[2:4]))
print(np.dot(B[1],res.x[2:4]))

L2 = -res.x[2]*t/res.x[3] + res.x[0]/res.x[3]
# plt.ylim([-0.5,  1.2])
# plt.xlim([-0.2, 1.2])
# plt.plot(t,L2)

print("Classified in class B ",B[0])
print("Classified in class B ",B[1])
print("Parameters = ",res.x)

print("Iteration 3")

A = [[0,0]]

lhs_ineq = [[1, 0, 0, 0, 0],
            [0, 0, -1, 0, -Big],
            [0, 0, 0, -1, -Big],
            [0, 0, 1, 0, Big],
            [0, 0, 0, 1, Big]]

b_u    =   [ 0,
             -eps,
             -eps,
             Big -eps,
             Big -eps]
b_l = np.full_like(b_u, -np.inf)
constraints = LinearConstraint(lhs_ineq, b_l, b_u)

res = milp(c=obj, constraints=constraints, integrality=integr,bounds=Bnd)

print(np.dot(A[0],res.x[2:4]))

print("Classified in class A ",A[0] )
print("Parameters = ",res.x)

L2 = -res.x[2]*t/res.x[3] + 50/res.x[3]
plt.plot(t,L2)