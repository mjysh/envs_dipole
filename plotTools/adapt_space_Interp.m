function [UInterp,VInterp,OInterp] = adapt_space_Interp(posX,posY,UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX)
rx = -1;
ry = -1;
regionN = length(UUU);
for i = 1:regionN
    if posX>=XMIN{i} && posX<XMAX{i} && posY>=YMIN{i} && posY<YMAX{i}
        Uf = UUU{i};
        Vf = VVV{i};
        Of = OOO{i};
        dx = (XMAX{i}-XMIN{i})/(width(Uf)-1);
        dy = (YMAX{i}-YMIN{i})/(height(Uf)-1);
        indexX = ceil((posX-XMIN{i})/dx);
        rx = mod((posX-XMIN{i}), dx);
        indexY = ceil((posY-YMIN{i})/dy);
        ry = mod((posY-YMIN{i}), dy);
        break
    end
end

if rx == -1
    posX, posY
    disp('cannot be reached');
    XMIN, XMAX, YMIN, YMAX
end
xa = [1-rx/dx, rx/dx];
ya = [1-ry/dy; ry/dy];
%     #########################################################
%     """interpolate velocity"""
QU = Uf(indexY:indexY+1,indexX:indexX+1)';
QV = Vf(indexY:indexY+1,indexX:indexX+1)';
QO = Of(indexY:indexY+1,indexX:indexX+1)';
UInterp = xa * QU * ya;
VInterp = xa * QV * ya;
OInterp = xa * QO * ya;
end