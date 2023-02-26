""" dipole """
bl = 1               # length                               
bw = 0.2             # width
##===============================================================================================##
""" observations/actions/reward """
flowMode = 'reduced'                          # choice of flow environment, 'CFD','CFDwRot','reduced'
obsMode = 'egoLRGradOnlyNoVision'             # choice of observations, 'labframeOneSensor','egoOneSensor','egoOneSensorPlusOrt','egoTwoSensorLR','egoTwoSensorFB'
rewardMode = 'sourceseeking'
resetMode = 'Sourceseeking'
##===============================================================================================##
dt = 0.1
mu = 0.8                                  # swimming speed
flexibility = 0.5                         # amount of change allowed in vortex strength

train_offset = 0.15                        # gap distance between training and target areas
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
period = lam/0.6096427823203662
##===============================================================================================##