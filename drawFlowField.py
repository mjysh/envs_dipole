#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 17:28:56 2022

@author: yusheng
"""
import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import CFDfunctions as cf

boundL = -23.75
boundR = 7.75

boundU = 7.75
boundD = -7.75
plot_dpi = 240

#%%
paramSource = 'envParam_ego2sensorLRGradCFD_200'
param = importlib.import_module('settings.'+paramSource)
cfdpath = param.cfdpath

time_span = param.time_span
level_limit = param.level_limit
source_path = cfdpath + "np/"
print('begin reading CFD data')
cfd_framerate,time_span,\
UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX\
= cf.adapt_load_data(time_span,source_path,level_limit)
print('finished reading CFD data')
#%%

nx = int(np.round((boundR - boundL)/0.03125)) + 1
ny = int(np.round((boundU - boundD)/0.03125)) + 1
x = np.linspace(boundL, boundR, nx)
y = np.linspace(boundD, boundU, ny)
X, Y = np.meshgrid(x, y, indexing = 'ij')
Omega = np.zeros_like(X)
# U = np.zeros_like(X)
# V = np.zeros_like(X)
# cmap = cm.get_cmap('bwr',128)
# index = 0
for t in np.arange(0, time_span+0.05, 0.05):
    print('time', t)
    for i in range(X.shape[0]):
        # print('x =', X[i,0])
        for j in range(X.shape[1]):
            # posx = (X[i,j] + X[i+1,j])/2
            # posy = (Y[i,j] + Y[i,j+1])/2
            posx = -X[i,j]
            posy = -Y[i,j]

            _,_,Omega[i,j] =  cf.adapt_time_interp(UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX,cfd_framerate,\
                                               time = t,posX = posx,posY = posy)
        # print(o)
    # plot the vorticity field
    fig, ax = plt.subplots(dpi = plot_dpi)
    fig.set_figwidth(8)
    # ax.quiver(X,Y,-U,-V)
    ax.pcolormesh(X,Y,Omega,shading = 'gouraud',cmap = cm.get_cmap('bwr',300),vmin = -3, vmax = 3)
    ax.set_xlim(left = -24, right = 8)
    ax.set_ylim(bottom = -8, top = 8)
    ax.set_aspect('equal')
    ax.axis('off')
    index = int(np.round(t/0.05))
    fig.savefig(f'/home/yusheng/cylinder_flow/Re=200/Movie/movie{index:04d}.png',pad_inches=0, bbox_inches='tight')
    plt.close(fig)
    # index += 1
    
