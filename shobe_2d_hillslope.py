# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:23:41 2016

@author: Charlie

2-dimensional diffusive hillslope evolution.
Solved with the Peaceman-Rachford 
Alternating Direction Implicit (ADI)
method, an unconditionally stable,
second order accurate predictor-
corrector scheme solved usint the Thomas algorithm.
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import pylab
from mpl_toolkits.mplot3d import Axes3D

def pyThomas(a,b,c,r): #MODIFIED TO TRY TO MAKE ADI SOLUTION WORK
    '''
    Solves tridiagonal system of equations, in this case for the purpose of
    implicit solutions for numerical models.
    '''
    J = len(b) #number of nodes
    #j = np.arange(1,J) #node vector

    f = np.zeros(len(b))
    g = np.zeros(len(b))
    f[0] = b[0]
    g[0] = c[0] / f[0]
    
    e = a #assumes zero has already been added by code that calls Thomas

    for i in range(1,J):
        f[i] = b[i] - e[i] * g[i-1]
        g[i] = c[i] / f[i]
    y = np.zeros(len(b))
    y[0] = r[0] / f[0]
    for i2 in range(1, J):
        y[i2] = (r[i2] - e[i2] * y[i2-1]) / f[i2]
    
    x = np.zeros(len(b))
    x[J-1] = y[J-1]
    for i3 in range(J-1, 0, -1): #iterate backwards through nodes
        x[i3-1] = y[i3-1] - g[i3-1] * x[i3]

    return x
    
initial_flat_height = 1000 #m
#uplift_rate = 0.001 #m yr-1
k = 1 #m yr-1 m-2
rho_soil = 1500 #kg m-3
x_min = 0 #meters
dx = 1 #meters
x_max = 10 #meters
nx = x_max / dx

y_min = 0 #meters
dy = dx #meters, forces square grid cells
y_max = 10 #meters
ny = y_max / dy

t_min = 0 #years
dt = 1 #years
t_max = 1000 #years

x = np.arange(x_min, x_max, dx)
y = np.arange(y_min, y_max, dy)
times = np.arange(t_min, t_max, dt)

z_initial = np.zeros((len(y), len(x))) #elevation grid
z_initial[:, :] = initial_flat_height
kappa = k / rho_soil #soil diffusivity, m yr-1

rx = kappa * dt / (dx ** 2)
ry = kappa * dt / (dy ** 2)

#boundary condition arrays: for now, constant baselevel drop
bc_up = np.arange(1000, 0, -.001)
bc_down = np.arange(1000, 0, -.001)
bc_left = np.arange(1000, 0, -.001)#np.zeros((len(bc_up)))
bc_right = np.arange(1000, 0, -.001)#np.zeros((len(bc_up)))
#bc_left[:] = 1000
#bc_right[:] = 1000
z_old = z_initial
it = 0
z_predictor = np.zeros((len(y), len(x)))
z_corrector = np.zeros((len(y), len(x)))

c1 = 1
c2 = 0
c3 = 1
c4 = 0

#pre-allocate predictor jacobian
#works as long as dx==dy, nx==ny, rx==ry
a_guts = np.repeat(-rx / 2, ny-2)
a_add_elem1 = np.insert(a_guts, 0, 0)
a_final = np.append(a_add_elem1, -c4/dx) #only works with Dirichlet boundary
b_guts = np.repeat(1 + rx, ny-2)
b_add_elem1 = np.insert(b_guts, 0, (c1 - c2 / dx))
b_final = np.append(b_add_elem1, (c3 + c4 / dx))
c_guts = np.repeat(-rx / 2, ny-2)
c_add_elem1 = np.insert(c_guts, 0, -c2 / dx) #only works with Dirichlet boundary
c_final = np.append(c_add_elem1, 0)

for t in range(len(times)-1):
    it += 1
    #print it
    current_time = times[t] + dt
    print current_time
    for i in range(1, int(ny) - 1): #predictor step needs to do an inversion for every interior node row
        r_guts = z_old[i, 1:-1] + ry * (z_old[i, 0:-2] - 2 * z_old[i, 1:-1] + z_old[i, 2:]) / 2
        r_temp = np.insert(r_guts, 0, bc_left[it - 1])
        r = np.append(r_temp, bc_right[it - 1])    
        z_predictor_row = pyThomas(a_final, b_final, c_final, r)
        z_predictor[i, :] = z_predictor_row
    z_predictor[0, :] = bc_up[it - 1]
    z_predictor[-1, :] = bc_down[it - 1]
        
    for j in range(1, int(nx) - 1): #corrector needs to do an inversion for every column
        r_guts = z_predictor[1:-1, j] + rx * (z_predictor[0:-2, j] - 2 * z_predictor[1:-1, j] + z_predictor[2:, j]) / 2
        r_temp = np.insert(r_guts, 0, bc_up[it - 1])
        r = np.append(r_temp, bc_down[it - 1])    
        z_corrector_col = pyThomas(a_final, b_final, c_final, r)
        z_corrector[:, j] = z_corrector_col
    z_corrector[:, 0] = bc_left[it - 1]        
    z_corrector[:, -1] = bc_right[it - 1] 
    z_corrector[0, :] = bc_up[it - 1]
    z_corrector[-1, :] = bc_down[it - 1]
    z_old = z_corrector

x, y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z_old, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
plt.show()