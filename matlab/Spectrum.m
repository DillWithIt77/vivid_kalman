function [ENE, ENS] = Spectrum(q_hat,p)
% Function takes Fourier coefficients of PV (q_hat) and struct containing
% parameters (p) and evaluates energy and enstrophy 1D spectra
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k = [0:p.N/2 -p.N/2+1:-1]';
[KX, KY] = meshgrid(k,k);
dX = 1i*repmat(k',[p.N 1]);
dY = 1i*repmat(k,[1 p.N]);
Laplacian = dX(:,:).^2+dY(:,:).^2;
InvBT = 1./Laplacian; InvBT(1,1) = 0;

% Invert for psi
q_bt = q_hat(:,:); % + hk(:,:);
psi_hat = InvBT.*q_hat;


ENE = zeros(p.N/2+1,1);
ENS = zeros(p.N/2+1,1);
for jj = 1:p.N
    for ii = 1:p.N
        k = sqrt(KX(ii,jj).^2 + KY(ii,jj).^2);
        if ceil(k) <= p.N/2
            r = k-floor(k);
            % plus 1 because of matlab indexing! 
            ENE(floor(k)+1) = ENE(floor(k)+1) + (1-r)*(k^2)*abs(psi_hat(ii,jj))^2;
            ENS(floor(k)+1) = ENS(floor(k)+1) + (1-r)*abs(q_bt(ii,jj))^2;
            ENE(ceil(k)+1) = ENE(ceil(k)+1) + r*(k^2)*abs(psi_hat(ii,jj))^2;
            ENS(ceil(k)+1) = ENS(ceil(k)+1) + r*abs(q_bt(ii,jj))^2;
        end
    end
end
ENE = .5*ENE/(p.N^4);
ENS = .5*ENS/(p.N^4);
