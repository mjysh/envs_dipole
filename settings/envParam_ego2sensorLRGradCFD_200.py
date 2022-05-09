""" dipole """
bl = 1               # length                               
bw = 0.2             # width
##===============================================================================================##
""" observations/actions/reward """
flowMode = 'CFD'                          # choice of flow environment, 'CFD','CFDwRot','reduced'
obsMode = 'egoTwoSensorLRGrad'             # choice of observations, 'labframeOneSensor','egoOneSensor','egoOneSensorPlusOrt','egoTwoSensorLR','egoTwoSensorLRGrad','egoTwoSensorFB'
##===============================================================================================##
dt = 0.1
mu = 0.8                                  # swimming speed
flexibility = 0.5                         # amount of change allowed in vortex strength
train_offset = 0.15                        # gap distance between training and testing areas
cfdpath = '/home/yusheng/cylinder_flow/Re=200/'

##===============================================================================================##
"""  for reduced-order wake """
# permitted range of area
reducedDomainL = -8
reducedDomainR = 8
reducedDomainU = 5.5
reducedDomainD = -5.5
A = 0.5
lam = 3
Gamma = 3
bgflow = -1.0                             # background horizontal flow velocity
cut = 0.5

##===============================================================================================##
#  for CFD
# permitted range of area
cfdDomainL = 0.5
cfdDomainR = 23.5
cfdDomainU = 6
cfdDomainD = -6
time_span = 5                 # maximum simulation time
level_limit = 3                # max level of CFD grids, higher level means higher precision
