""" dipole """
bl = 1               # length                               
bw = 0.2             # width
##===============================================================================================##
""" observations/actions/reward """
flowMode = 'reduced'                          # choice of flow environment, 'CFD','CFDwRot','reduced'
obsMode = 'egoDirLRGrad'             # choice of observations, 'labframeOneSensor','egoOneSensor','egoOneSensorPlusOrt','egoTwoSensorLR','egoTwoSensorLRGrad','egoTwoSensorFB'
##===============================================================================================##
dt = 0.1
mu = 0.8                                  # swimming speed
flexibility = 0.5                         # amount of change allowed in vortex strength
train_offset = 0.15                        # gap distance between training and testing areas
cfdpath = '/home/yusheng/CFDadapt/'

##===============================================================================================##
"""  for reduced-order wake """
# permitted range of area
reducedDomainL = -30
reducedDomainR = 6
reducedDomainU = 12
reducedDomainD = -12
A = 0.2
lam = 4
Gamma = 3
bgflow = -1.0                             # background horizontal flow velocity
cut = 0.6
period = lam/0.6096427823203662
##===============================================================================================##
#  for CFD
# permitted range of area
cfdDomainL = -23.5
cfdDomainR = -0.5
cfdDomainU = 6
cfdDomainD = -6
time_span = 4.5                 # maximum simulation time (one wake period)
level_limit = 3                # max level of CFD grids, higher level means higher precision