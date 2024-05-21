""" dipole """
bl = 1               # length                               
bw = 0.2             # width
##===============================================================================================##
""" observations/actions/reward """
flowMode = 'reduced'                          # choice of flow environment, 'CFD','CFDwRot','reduced'
obsMode = 'egoTwoSensorLRGrad'             # choice of observations, 'labframeOneSensor','egoOneSensor','egoOneSensorPlusOrt','egoTwoSensorLR','egoTwoSensorLRGrad','egoTwoSensorFB'
##===============================================================================================##
dt = 0.1
mu = 0.8                                  # swimming speed
flexibility = 0.5                         # amount of change allowed in vortex strength
train_offset = 0.15                        # gap distance between training and testing areas
cfdpath = '/home/yusheng/CFDadapt/'

##===============================================================================================##
"""  for reduced-order wake """
# permitted range of area
reducedDomainL = -24
reducedDomainR = 0
reducedDomainU = 6
reducedDomainD = -6
A = 0.2
lam = 4
Gamma = 3
bgflow = -1.0                             # background horizontal flow velocity
cut = 0.6
import numpy as np
period = lam/(np.abs(bgflow)- Gamma/2/lam*np.tanh(2*np.pi*A/lam))
##===============================================================================================##
#  for CFD
# permitted range of area
cfdDomainL = -23.5
cfdDomainR = -0.5
cfdDomainU = 6
cfdDomainD = -6
time_span = 4.5                 # maximum simulation time (one wake period)
level_limit = 3                # max level of CFD grids, higher level means higher precision