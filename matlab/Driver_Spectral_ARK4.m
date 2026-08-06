function datafolder = Driver_Spectral_ARK4(N,beta,kf,epsf,sigf,alpf, dd,tau0,tdel,Nt,forcingtype)
% This script solves barotropic QG flow in a doubly-periodic domain.
% adaptive RK4 integrator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set simulation parameters
% N  % Number of points in each direction, N dividable by 4
% dt = 0.01; % initial time step size
% Nt % Number of time steps
qlim = 1.5E5; % if any q > qlim, simulation stops

% Set physical parameters
% epsf forcing amplitude
% dd  % linear Ekman damping

% tau0  % surface wind stress strength
% tdel  % width of wind stress

sh = 8; % hyperdiffusion exponent (needs to be even)
if N == 128
    if epsf == 0
        nu = 2.5e-28 *(50/50)^(-2*sh+2/3); %5E-33; % Coefficient of biharmonic vorticity diffusion
    else
        nu = 2.5e-28 *(kf/50)^(-2*sh+2/3); %5E-33; % Coefficient of biharmonic vorticity diffusion
    end
elseif N == 256
    nu = 5E-33 *(kf/50)^(-2*sh+2/3); %5E-33; % Coefficient of biharmonic vorticity diffusion
    if tau0 == 0
        nu = nu*100;
    end
elseif N == 512
    nu = 1E-34 *(kf/50)^(-2*sh+2/3);
end


% Put useful stuff into a struct
params = struct('beta',beta, 'sh',sh, 'dd',dd, 'nu',nu, 'N',N,...
    'kf',kf, 'epsf',epsf, 'sigf',sigf, 'alpf',alpf,...
    'tau0',tau0, 'tdel',tdel,'forcingtype',forcingtype);


% Set up hyperviscous PV dissipation and linear damping
k = [0:N/2 -N/2+1:-1]'; % wavenumbers
L = zeros(N,N);
for jj=1:N
    for ii=1:N
        kr = sqrt(k(ii)^2+k(jj)^2);
        L(ii,jj) = -nu*kr^(2*sh);
    end
end
% Set up stochastic forcing with spectral
Fk = zeros(N,N);
for ii=1:N
    for jj=1:N
        kr = sqrt(k(ii)^2+k(jj)^2);
        phi=atan(k(jj)/k(ii));
        if isnan(phi)
            phi=0;
        end
        ampl = epsf*(1+alpf*cos(2*phi));
        Fk(ii,jj) = (ampl*kr)*exp(-.5*(kr-kf)^2/sigf^2)/(sqrt(2*pi)*sigf);
    end
end
Fk(1,1)=0;
clear kr ii jj

% surface wind stress
dx=2*pi/N;
[KX, KY] = meshgrid(k,k);
[X,Y] = meshgrid(-pi:dx:pi-dx,-pi:dx:pi-dx);

if strcmp(forcingtype,'sin')
    tau = tau0 *sin(Y).^2;
elseif strcmp(forcingtype,'sinh')
    tau = tau0 *sech(Y/tdel).^2;
elseif strcmp(forcingtype,'constant')
    tau = tau0;
end
tauk = -1i*KY.*fft2(tau);
clear k KX KY


% Initialize
t = 0;
qp(:,:) = 0.01*randn(params.N);
qp(:,:) = qp(:,:) - mean(mean(qp(:,:)));
q = fft2(qp);

% Initialize topography (no topography)
% dx=2*pi/N;
% [X,Y]=meshgrid(-pi:dx:pi-dx,-pi:dx:pi-dx);
% topo = exp(-(X.^2+Y.^2)/2/1^2); %sin(2*X).*sin(2*Y);
% topo = topo-mean(mean(topo));
% global hk
% hk = fft2(topo);

% Diagnostics
countDiag = 100; % Compute diagnostics every countDiag steps
T = zeros(1, Nt/countDiag);
vb = zeros(1, Nt/countDiag);
utz = zeros(N, Nt/countDiag);
energy = zeros(N/2+1, Nt/countDiag);
enstrophy = zeros(N/2+1, Nt/countDiag);
dtsize = zeros(1, Nt/countDiag);

qk1 = zeros(N,N); % mean
qk2 = zeros(N,N); % var
count = 0;

% Data folder
if tdel == 0
    fstr = sprintf('forcing-%s_tau0-%.2g', forcingtype, tau0);
else
    fstr = sprintf('forcing-%s_tau0-%.2g_tdel%.2g', forcingtype, tau0, tdel);
end
if epsf ~= 0
    if alpf == 0
        fstr = sprintf('%s_kf%i_epsf%0.2g_sigf%0.1g', fstr, kf, epsf, sigf);
    else
        fstr = sprintf('%s_kf%i_epsf%0.2g_sigf%0.1g_alpha%0.0g', fstr, kf, epsf, sigf, alpf);
    end
end  
casestr = sprintf('baroARK4_N%i_beta%0.2g_d%.2g_nu%.2g_%s_Nt%i', N, beta, dd, nu, fstr, Nt);

casestr = strcat(casestr, strcat('_set-', datestr(datetime('now'),'mm-dd-yy-hhMMss')));

datafolder = fullfile('data', casestr);
if ~isdir(datafolder); mkdir(datafolder); end

% Save data
qh_dataname = fullfile(datafolder, 'qh.bin');
time_dataname = fullfile(datafolder, 't.bin');

fid1 = fopen(qh_dataname,'w');
fid2 = fopen(time_dataname,'w');

fwrite(fid1, [real(q(:)) imag(q(:))], 'double');
fwrite(fid2, t, 'double');
    
% adaptive stepping stuff:
tol= 1E-1;
r0 = 0.8*tol;

dt = 1E-5; % initial time step size

% Main loop
for ii=1:Nt
    if mod(ii, countDiag) == 0
        if any(isnan(q(:))), break, end
        T(ii/countDiag) = t;
        [ENE,ENS] = Spectrum(q, params);
        dtsize(ii/countDiag) = dt;
        energy(:, ii/countDiag) = ENE;

        enstrophy(:, ii/countDiag) = ENS;
        [VB,UTZ] = QG_Diagnostics(q, params);
        vb(ii/countDiag) = VB; utz(:, ii/countDiag) = UTZ;
        if ii > Nt/2
            count = count+1;
            qk1 = qk1+q;
            qk2 = qk2+abs(q).^2;
        end
        if mod(ii, 1e4) == 0
            save(fullfile(datafolder,'diagnostics.mat'),...
                'ii','countDiag','dt','params','T','energy','enstrophy','vb','utz','qp', 'X','Y','qk1','qk2','count','dtsize','-v7.3');
        end
        fprintf('iteration %i\n', ii);
    end
    
    M = 1./(1-.25*dt*L);
    % First stage ARK4
    k0 = RHS_Spectral(q,params) + tauk;
    l0 = L.*q;
    % Second stage
    q1 = M.*(q+.5*dt*k0+.25*dt*l0);
    k1 = RHS_Spectral(q1,params) + tauk;
    l1 = L.*q1;
    % Third stage
    q2 = M.*(q+dt*(13861*k0/62500+6889*k1/62500+8611*l0/62500-1743*l1/31250));
    k2 = RHS_Spectral(q2,params) + tauk;
    l2 = L.*q2;
    % Fourth stage
    q3 = M.*(q+dt*(-0.04884659515311858*k0-0.1777206523264010*k1+0.8465672474795196*k2...
        +0.1446368660269822*l0-0.2239319076133447*l1+0.4492950415863626*l2));
    k3 = RHS_Spectral(q3,params) + tauk;
    l3 = L.*q3;
    % Fifth stage
    q4 = M.*(q+dt*(-0.1554168584249155*k0-0.3567050098221991*k1+1.058725879868443*k2...
        +0.3033959883786719*k3+0.09825878328356477*l0-0.5915442428196704*l1...
        +0.8101210538282996*l2+0.2831644057078060*l3));
    k4 = RHS_Spectral(q4,params) + tauk;
    l4 = L.*q4;
    % Sixth stage
    q5 = M.*(q+dt*(0.2014243506726763*k0+0.008742057842904184*k1+0.1599399570716811*k2...
        +0.4038290605220775*k3+0.2260645738906608*k4+0.1579162951616714*l0...
        +0.1867589405240008*l2+0.6805652953093346*l3-0.2752405309950067*l4));
    k5 = RHS_Spectral(q5,params) + tauk;
    l5 = L.*q5;
    
    % Error control %%%%%%%%%%%%%%%%%%%%%
    r1 = dt*max(max(max(abs(ifft2(0.003204494398459*(k0+l0) -0.002446251136679*(k2+l2)-0.021480075919587*(k3+l3)...
        +0.043946868068572*(k4+l4) -0.023225035410765*(k5+l5))))));
    if r1>tol, dt=.75*dt; continue, end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Stochastic forcing
    dW1 = randn(N,N/2+1);
    dW2 = randn(N,N/2+1);
    dW = (dW1+1i*dW2)/sqrt(2);
    dW(1,1) = 0;
    dW(1,N/2+1) = dW1(1,N/2+1);
    dW(N/2+1,1) = dW1(N/2+1,1);
    dW(N/2+1,N/2+1) = dW1(N/2+1,N/2+1);
    dW(N/2+2:end,1) = conj(dW(N/2:-1:2,1));
    dW(N/2+2:end,N/2+1) = conj(dW(N/2:-1:2,N/2+1));
    
    xik = N^2*[dW, conj(dW([1,N:-1:2],N/2:-1:2))].*sqrt(Fk);
    
    
    % Successful step, proceed to evaluation
    t = t + dt;
    qp = real(ifft2( ...
        q + dt*(0.1579162951616714*(k0+l0)+0.1867589405240008*(k2+l2)+...
        0.6805652953093346*(k3+l3)-0.2752405309950067*(k4+l4)+(k5+l5)/4)...
        + sqrt(dt)*xik ));
    q = fft2(qp);
    
    
    % Save data
    fwrite(fid1, [real(q(:)) imag(q(:))], 'double');
    fwrite(fid2, t, 'double');

    
    % step size adjustment: EPS, PI.3.4 ; divide by 4 for a 4th order
    % method with 3rd order embedded %%%%%%%%%%%%%%%%%%%%%
    dt = ((.75*tol/r1)^(.3/4))*((r0/r1)^(.4/4))*dt;
    r0 = r1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    if any(abs(qp(:))>qlim), break, end
end

if any(isnan(q(:)))
    fprintf('NaN solution detected!\n')
else
    save(fullfile(datafolder,'diagnostics.mat'),...
                'ii','countDiag','dt','params','T','energy','enstrophy','vb','utz','qp', 'X','Y','qk1','qk2','count','dtsize','-v7.3');
end


% status1 = fclose(fid1);
% status2 = fclose(fid2);
% if status1 || status2
%     warn('fclose not sucessful');
% end

fprintf('done solver');