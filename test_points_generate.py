#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:49:13 2022

@author: yusheng
"""
import numpy as np
N = 200
x = np.random.rand(N,3)
y = np.random.rand(N,2)

positions = np.zeros_like(x)
swimmer_centerX, swimmer_centerY = -12, -2.15
rmax = 2
j = 0
for a,b,c in x:
    r = rmax*np.sqrt(a)
    the = b*2*np.pi
    positions[j,0] = swimmer_centerX + np.cos(the)*r
    positions[j,1] = swimmer_centerY + np.sin(the)*r
    positions[j,2] = c*2*np.pi
    j += 1

targets = np.zeros_like(y)
target_centerX, target_centerY = -12, 2.15
rmax = 2
j = 0
for a,b in y:
    r = rmax*np.sqrt(a)
    the = b*2*np.pi
    targets[j,0] = target_centerX + np.cos(the)*r
    targets[j,1] = target_centerY + np.sin(the)*r
    j += 1
