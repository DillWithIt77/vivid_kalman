function [vb,utz] = QG_Diagnostics(q_hat,p)
% Function takes Fourier coefficients of q and computes
% area-integrated meridional heat flux
%   vb := .5*kd^2*int((psi_2)_x psi_1)
% (Note that this equals barotropic v times baroclinic streamfunction up to
% the factor of kd^2.)
% and it returns zonally-averaged barotropic (utz) zonal velocity.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

persistent dX dY Laplacian InvBT KX KY
if isempty(dX)
    k = [0:p.N/2 -p.N/2+1:-1]';
    [KX, KY] = meshgrid(k,k);
    dX = 1i*repmat(k',[p.N 1]);
    dY = 1i*repmat(k,[1 p.N]);
    Laplacian = dX(:,:).^2+dY(:,:).^2;
    InvBT = 1./Laplacian; InvBT(1,1) = 0;
    
    k = [0:p.N/2-1 0 -p.N/2+1:-1]';
    dX = 1i*repmat(k',[p.N 1]);
    dY = 1i*repmat(k,[1 p.N]);
    clear k
end

% Invert for psi
q_bt = q_hat(:,:); %+hk(:,:);
psi_hat = InvBT.*q_hat;

% Real-Space quantities
vt = real(ifft2(dX.*psi_hat));
ut = real(ifft2(-dY.*psi_hat));

% Outputs
vb  = ((2*pi/p.N)^2)*sum(sum(vt.*q_bt));
utz = mean(ut,2);