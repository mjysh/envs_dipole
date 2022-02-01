""" dipole """
bl = 1               # length                               
bw = 0.2             # width
##===============================================================================================##
""" observations/actions/reward """
flowMode = 'reduced'                          # choice of flow environment, 'CFD','CFDwRot','reduced'
obsMode = 'egoTwoSensorLRGradOnly'             # choice of observations, 'labframeOneSensor','egoOneSensor','egoOneSensorPlusOrt','egoTwoSensorLR','egoTwoSensorLRGrad','egoTwoSensorFB'
##===============================================================================================##
dt = 0.1
mu = 0.8                                  # swimming speed
flexibility = 0.5                         # amount of change allowed in vortex strength
train_offset = 0.15                        # gap distance between training and testing areas
cfdpath = '/home/yusheng/CFDadapt/'

##===============================================================================================##
"""  for reduced-order wake """
# permitted range of area
reducedDomainL = -8
reducedDomainR = 8
reducedDomainU = 5.5
reducedDomainD = -5.5
A = 0.3
lam = 4
Gamma = 3
bgflow = -1.0                             # background horizontal flow velocity
cut = 0.6

##===============================================================================================##
#  for CFD
# permitted range of area
cfdDomainL = -15.5
cfdDomainR = 7.5
cfdDomainU = 5.5
cfdDomainD = -5.5
time_span = 60                 # maximum simulation time
level_limit = 3                # max level of CFD grids, higher level means higher precision
