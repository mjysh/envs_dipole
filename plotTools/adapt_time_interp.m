function [U_interp,V_interp,O_interp] = adapt_time_interp(CFD,time,posX, posY)
tn = length(CFD.UUU);
time = mod(time,CFD.time_span);
frame = time*CFD.cfd_framerate;
if frame > tn-1
    frameUp = tn - 1;
    frameDown = tn - 1;
elseif frame < 0
    frameUp = 0;
    frameDown = 0;
else
    frameUp = ceil(frame);
    frameDown = floor(frame);
end
weightDown = (frameUp-frame);
weightUp = (frame-frameDown);
%     """velocity for frameDown"""
k = frameDown + 1;
[UDown,VDown,ODown] = adapt_space_Interp(posX,posY,CFD.UUU{k},CFD.VVV{k},CFD.OOO{k},CFD.XMIN{k},CFD.XMAX{k},CFD.YMIN{k},CFD.YMAX{k});
%     # #########################################################
if frameDown ~= frameUp
    %         """velocity for frameUp"""
    k = frameUp + 1;
    [UUp,VUp,OUp] = adapt_space_Interp(posX,posY,CFD.UUU{k},CFD.VVV{k},CFD.OOO{k},CFD.XMIN{k},CFD.XMAX{k},CFD.YMIN{k},CFD.YMAX{k});
    U_interp = UUp*weightUp + UDown*weightDown;
    V_interp = VUp*weightUp + VDown*weightDown;
    O_interp = OUp*weightUp + ODown*weightDown;
else
    U_interp = UDown+0;
    V_interp = VDown+0;
    O_interp = ODown+0;
end
end