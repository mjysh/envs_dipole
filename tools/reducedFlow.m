function [flowU,flowV] = reducedFlow(states,time)
    A = 0.2;
    Gamma = 3;
    lam = 4;
    z = states(1)+1j*states(2);
    U = Gamma/2/lam*tanh(2*pi*A/lam) - 1;
    cut = 0.6;
    
    singular1 = tan(pi*(z + 1j*A - time*U)/lam);
    singular2 = tan(pi*(z-lam/2-1j*A - time*U)/lam);
    if (abs(singular1) > 1e-3) && (abs(singular2) > 1e-3)
        wVK = 1j*Gamma/2/lam*(1/singular1 - 1/singular2);
    else
        wVK = 0;
    end
    flowU = real(wVK);
    flowV = -imag(wVK);
    if flowU > cut
        flowU = cut;
    elseif flowU < -cut
        flowU = -cut;
    end
    if flowV > cut
        flowV = cut;
    elseif flowV < -cut
        flowV = -cut;
    end
    flowU = flowU - 1;
    % vorticity = 0;
end