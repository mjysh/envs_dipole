""" dipole """
bl = 1               # length                               
bw = 0.2             # width
##===============================================================================================##
""" observations/actions/reward """
flowMode = 'CFD'                          # choice of flow environment, 'CFD','CFDwRot','reduced'
obsMode = 'EgoVelMagGrad'             # choice of observations, 'labframeOneSensor','egoOneSensor','egoOneSensorPlusOrt','egoTwoSensorLR','egoTwoSensorFB'
sensorLocation = -0.9
##===============================================================================================##
dt = 0.1
mu = 0.25                                  # swimming speed
angularSpeed = 3 	           	 	# amount of change allowed in vortex strength
cfdpath = '/home/yusheng/CFDadapt/'

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
cfdDomainL = -22.5
cfdDomainR = 0.5
cfdDomainU = 6.5
cfdDomainD = -6.5
time_span = 0.5                 # maximum simulation time
level_limit = 3                # max level of CFD grids, higher level means higher precision
