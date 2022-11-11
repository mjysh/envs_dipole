function [UInterp,VInterp,OInterp] = adapt_space_Interp(posX,posY,UUU,VVV,OOO,XMIN,XMAX,YMIN,YMAX)
rx = -1;
ry = -1;
regionN = length(UUU);
for i = 1:regionN
    dx = (XMAX{i}-XMIN{i})/(width(UUU{i})-1);
    dy = (YMAX{i}-YMIN{i})/(height(VVV{i})-1);
    if posX>=XMIN{i}-dx/2 && posX<XMAX{i}+dx/2 && posY>=YMIN{i}-dy/2 && posY<YMAX{i}
        Uf = UUU{i};
        Vf = VVV{i};
        Of = OOO{i};
        indexX = ceil((posX-XMIN{i})/dx);
        indexY = ceil((posY-YMIN{i})/dy);

        if indexX < 1
            %                 extrapolation
            rx = posX - XMIN{i};
            indexX = 1;
        elseif indexX > width(Uf) - 1
            %                 extrapolation
            rx = posX - XMAX{i} + dx;
            indexX = width(Uf) - 1;
        else
            %                 interpolation
            rx = mod((posX - XMIN{i}), dx);
        end
        if indexY < 1
            %                 extrapolation
            ry = posY - YMIN{i};
            indexY = 1;
        elseif indexY > height(Uf) - 1
            %                 extrapolation
            ry = posY - YMAX{i} + dy;
            indexY = height(Uf) - 1;
        else
            %                 interpolation
            ry = mod((posY - YMIN{i}), dy);
        end
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