import importlib
import CFDfunctions as cf
import numpy as np
paramSource = 'envParam_ego2sensorLRGradCFD_300'
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
xlist = [n+0.45 for n in range(1,23)]
ylist = [0.25, 0, -0.25]
uVK0 = np.zeros((len(xlist)*len(ylist),1))
vVK0 = np.zeros((len(xlist)*len(ylist),1))
oVK0 = np.zeros((len(xlist)*len(ylist),1))
k = 0
t = 0
for x in xlist:
    for y in ylist:
        uVK0[k],vVK0[k],oVK0[k] =  cf.adapt_time_interp(UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX,cfd_framerate,\
                                    time = t,posX = x,posY = y)
        k += 1
error = np.zeros((len(UUU)-1,))

for i in range(1,len(UUU)):
    uVK = np.zeros((len(xlist)*len(ylist),1))
    vVK = np.zeros((len(xlist)*len(ylist),1))
    oVK = np.zeros((len(xlist)*len(ylist),1))
    k = 0
    t = time_span/(len(UUU)-1)*i
    print("*******************",t)
    for x in xlist:
        for y in ylist:
            uVK[k],vVK[k],oVK[k] =  cf.adapt_time_interp(UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX,cfd_framerate,\
                                        time = t,posX = x,posY = y)
            k += 1
    error[i-1] = np.linalg.norm(uVK-uVK0) + np.linalg.norm(vVK-vVK0) + np.linalg.norm(oVK-oVK0)
print(np.sort(error)[:6])
print(np.argsort(error)[:6]+1)