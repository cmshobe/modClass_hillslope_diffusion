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

Similar to Perron (2011), but this solves the linear diffusion equation.

USER GUIDE:
-Inputs are found below the pyThomas function definition.
-Call this script from the terminal window with 'python shobe_2d_hillslope.py'
-If there is an error, you likely don't have the required dependencies installed,
but all dependencies are standard with Anaconda Python.
-The method is dependent on grid symmetry: dx must == dy, nx must == ny
-This model will output a final figure, called 'shobe_hillslope.png'
"""
#import libraries
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

#set global plotting parameters
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['xtick.major.pad']='10'
matplotlib.rcParams.update({'figure.autolayout': False})

def pyThomas(a,b,c,r): #solves tridiagonal matrix efficiently
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

##########USER DEFINED PARAMETERS#############################################
t_report = 2000 #years, plot every __ years
k = 0.05 #m yr-1 m-1
rho_soil = 1500 #kg m-3

#spatial domain
x_min = 0 #meters
dx = .5 #meters
x_max = 10 #meters
y_min = 0 #meters
dy = dx #meters, forces square grid cells
y_max = 10 #meters

#temporal domain
t_min = 0 #years
dt = 50 #years
t_max = 100000 #years
times = np.arange(t_min, t_max + dt, dt)

#baselevel forcing
bl_drop_sinusoid = -0.001 * np.sin(2 * np.pi * times / 40000) - .001
##############################################################################
initial_flat_height = 1000 #m
nx = x_max / dx
ny = y_max / dy
x = np.arange(x_min, x_max, dx)
y = np.arange(y_min, y_max, dy)

z_initial = np.zeros((len(y), len(x))) #elevation grid
z_initial[:, :] = initial_flat_height
kappa = k / rho_soil #soil diffusivity, m2 yr-1

rx = kappa * dt / (dx ** 2)
ry = kappa * dt / (dy ** 2)

#boundary condition arrays: for now, constant baselevel drop
bc_up = np.zeros(len(bl_drop_sinusoid))
bc_down = np.zeros(len(bl_drop_sinusoid))
bc_left = np.zeros(len(bl_drop_sinusoid))
bc_right = np.zeros(len(bl_drop_sinusoid))
bc_up[0] = initial_flat_height
bc_down[0] = initial_flat_height
bc_left[0] = initial_flat_height
bc_right[0] = initial_flat_height
for i1 in range(1,len(bl_drop_sinusoid)):
    bc_up[i1] = bc_up[i1 - 1] + bl_drop_sinusoid[i1] * dt 
    bc_down[i1] = bc_down[i1 - 1] + bl_drop_sinusoid[i1] * dt 
    bc_left[i1] = bc_left[i1 - 1] + bl_drop_sinusoid[i1] * dt 
    bc_right[i1] = bc_right[i1 - 1] + bl_drop_sinusoid[i1] * dt 

#instantiate iteration variables
z_old = z_initial
it = 0
z_predictor = np.zeros((len(y), len(x)))
z_corrector = np.zeros((len(y), len(x)))

#boundary condition constants for Jacobian matrix
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

#instantiate plotting architecture
x, y = np.meshgrid(x, y)
fig = plt.figure(figsize = (8, 10))
gs = gridspec.GridSpec(4, 4)
bl_drop_forcing = fig.add_subplot(gs[0, :])
bl_drop_forcing.invert_yaxis()
ax = fig.add_subplot(gs[1:3, :], projection='3d')
relief_plot = fig.add_subplot(gs[-1, :])
plt.ion()
plt.show()
plot_iter = 0

#instantiate record-keeping array for relief
relief_record = np.zeros((len(times)))

for t in range(len(times)): #iterate through timesteps, solve in two half-steps
    it += 1
    current_time = times[t] + dt
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
    #apply boundary conditions
    z_corrector[:, 0] = bc_left[it - 1]        
    z_corrector[:, -1] = bc_right[it - 1] 
    z_corrector[0, :] = bc_up[it - 1]
    z_corrector[-1, :] = bc_down[it - 1]
    z_old = z_corrector
    z_old_plot = z_old - np.amin(z_old) #adjust z values for nice looking plot
    
    #calculate relief
    relief = np.amax(z_old) - np.amin(z_old)
    relief_record[it-1] = relief
    if current_time % t_report == 0: #if it's time to plot... plot
        plot_iter += 1
        ax.clear()
        bl_drop_forcing.clear()
        surf = ax.plot_surface(x, y, z_old_plot, rstride=1, cstride=1, cmap=plt.cm.gist_earth, linewidth=0, antialiased=False, vmin = 0, vmax = 100)
        ax.set_xlim3d(x_min, x_max)
        ax.set_ylim3d(y_min, y_max)
        ax.set_zlim3d(0, 100)   
        ax.set_xlabel('\n X Distance [m]', labelpad = 50)
        ax.set_ylabel('\n Y Distance [m]', labelpad = 20)
        ax.set_zlabel('\n Elevation [m]', labelpad = 20)
        ax.text2D(0, 0.95, 'Time [yrs]: %.1f' % current_time, transform=ax.transAxes)
        ax.grid(False)
        if plot_iter == 1:
            fig.colorbar(surf, ax = ax)
        else:
            pass
        bl_drop_forcing.plot(times, bl_drop_sinusoid, linewidth = 2, color = 'k')
        bl_drop_forcing.plot(current_time, bl_drop_sinusoid[it - 1],marker = 'o', ms = 10, c = 'b')
        bl_drop_forcing.set_title('Forcing: Sinusoidal Baselevel Lowering Rate')
        bl_drop_forcing.set_xlabel('Time [yrs]')
        bl_drop_forcing.set_ylabel('Lowering Rate [m/yr]')
        relief_plot.plot(times[0:it-1], relief_record[0:it-1], linewidth = 2, color = 'r')
        relief_plot.set_xlim(0, t_max)
        relief_plot.set_ylim(0, 100)
        relief_plot.set_title('Hillslope Relief')
        relief_plot.set_xlabel('Time [yrs]')
        relief_plot.set_ylabel('Relief [m]')
        fig.subplots_adjust(left = 0.2, bottom = 0.15, hspace = 1.25)
        plt.pause(0.01)
    else:
        pass
fig.savefig('shobe_hillslope.png') #save final model figure