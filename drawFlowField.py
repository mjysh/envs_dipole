#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 17:28:56 2022

@author: yusheng
"""

import numpy as np
import matplotlib.pyplot as plt
from CFDfunctions import *

boundL = -8
boundR = 24
boundU = 8
boundD = -8

period = 4.985

for t in np.arange(0, 0.05, period+0.05):
    print('time', t)
    
