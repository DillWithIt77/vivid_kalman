clear;


fsavefigs = false;


casename = 'baroARK4_N128_beta0.05_d0.005_nu2.5e-28_forcing-sin_tau0-0.04_Nt10000_set-07-13-26-120015';

    
datafolder = fullfile('data', char(casename));
dataext = load(fullfile(datafolder, 'diagnostics.mat'));

figfolder = fullfile('figs', char(casename));
if fsavefigs && ~isdir(figfolder); mkdir(figfolder); end


%%
[qhc, tsave] = coarsenqh(datafolder, 0); % load data, second argument is junk
dt = tsave(2) - tsave(1);

nt = size(qhc,2);

N = sqrt(size(qhc,1));
k = [0:N/2 -N/2+1:-1]';
[KX, KY] = meshgrid(k,k);
Knorm = hypot(KX,KY);
Knorminv = 1./Knorm; Knorminv(1,1) = 0;

vh = -Knorminv(:).*qhc/N^2;

modes = zeros(4,nt); % save 1,0; 0,1; 1,1; -1,1 modes in order
% qlmodesa = zeros(4,nt); % save 1,0; 0,1; 1,1; -1,1 modes in order

modes(1,:) = vh(sub2ind([N,N],1,2),:); % 1,0
modes(2,:) = vh(sub2ind([N,N],2,1),:); % 0,1
modes(3,:) = vh(sub2ind([N,N],2,2),:); % 1,1
modes(4,:) = vh(sub2ind([N,N],2,N),:); % -1,1

% conjugate modes (not needed)
% qlmodesa(1,:) = vh(1,N); % 1,0
% qlmodesa(2,:) = vh(N,1); % 0,1
% qlmodesa(3,:) = vh(N,N); % 1,1
% qlmodesa(4,:) = vh(N,2); % -1,1

%%
handle = figure(1);
clf;
labels = ["(1,0)", "(0,1)", "(1,1)", "(-1,1)"];
for i = 1:size(modes,1)
    subplot(4,1,i)
    [Rmm,lags] = xcorr((modes(i,1:end).'- mean(modes(i,1:end),2)),'coeff');
    Rmm = Rmm(lags>0);
    lags = lags(lags>0);
    plot(lags*dt,real(Rmm)); hold on;
%     plot(lags*dt,imag(Rmm))
    if i == 1 || i == 2
    xlim([0 5e3*dt])
    else
    xlim([0 5e3*dt]); 
    end
    title(sprintf('mode %s',labels(i)));
    xlabel('t');
end
set(gcf,'Position',[867 204 348 709]);


if fsavefigs
    resizefonts(handle);
    export_fig(fullfile(figfolder, sprintf('autocorr')), '-q101','-p.01','-m1','-pdf');
    savefig(handle, fullfile(figfolder, sprintf('autocorr.fig')))
end


