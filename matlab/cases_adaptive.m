clear;
% 
% Forward cascade

N = 128;
beta = 0.05;
dd = 5E-3;

epsf = 0;
kf = 50;  % no effect since epsf = 0
sigf = 1; % no effect since epsf = 0
alpf = 0; % no effect since epsf = 0
tau0 = 0.04;
tdel = 0; % no effect for sin forcing
forcing_type = 'sin';

Nt = 5e4;

F = Driver_Spectral_ARK4(N, beta, kf, epsf, sigf, alpf, dd, tau0, tdel, Nt, forcing_type);

% F(1) = parfeval(@Driver_Spectral_ARK4, 1,  N, beta, kf, epsf, sigf, alpf, dd, tau0, tdel, Nt, forcing_type);
% F(2) = parfeval(@Driver_Spectral_ARK4, 1,  N, 0, kf, epsf, sigf, alpf, dd, tau0, tdel, Nt, forcing_type);

% % Inverse cascade
% 
% N = 128 ;
% beta = 29;
% dd = 5E-3;
% 
% epsf = 0.05;
% kf = 30;
% sigf = 1;
% alpf = 0;
% forcing_type = 'constant';
% tau0 = 0;
% tdel = 0;  % no effect since forcing constant
% 
% Nt = 1000;
% 
% F(1) = parfeval(@Driver_Spectral_ARK4, 1,  N, beta, kf, epsf, sigf, alpf, dd, tau0, tdel, Nt, forcing_type);
% F(2) = parfeval(@Driver_Spectral_ARK4, 1,  N, 0, kf, epsf, sigf, alpf, dd, tau0, tdel, Nt, forcing_type);
% 
% 
% while ~all(arrayfun(@(x) strcmp(x, 'finished'), {F.State}))
%     pause(10)
%     F.Diary
% end

% fetchOutputs(F,UniformOutput=false);










