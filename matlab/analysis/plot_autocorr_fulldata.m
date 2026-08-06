clear;

fsavefigs = true;

casename = 'baroARK4_N128_beta0.05_d0.005_nu2.5e-28_forcing-sin_tau0-0.04_Nt50000_set-07-13-26-121958';

datafolder = fullfile('data',casename);
dataext = load(fullfile(datafolder, 'diagnostics.mat'));

figfolder = fullfile('figs', casename);
if fsavefigs && ~isdir(figfolder); mkdir(figfolder); end


%%
fnameqh = fullfile(datafolder, 'qh.bin');
fnamet = fullfile(datafolder, 't.bin');

fidqh = fopen(fnameqh,'r');
fidt = fopen(fnamet,'r');

t = fread(fidt,'double');
dt = t(2) - t(1);
nt = length(t);
N =  dataext.params.N;
Ntotal = N*N;


k = [0:N/2 -N/2+1:-1]';
[KX, KY] = meshgrid(k,k);
Knorm = hypot(KX,KY);
Kinvnorm = 1./Knorm; Kinvnorm(1,1) = 0;
dX = 1i*repmat(k', [N 1]);
dY = 1i*repmat(k, [1 N]);
Laplacian = dX.^2 + dY.^2;
InvBT = 1./Laplacian; InvBT(1,1) = 0;

L = 2*pi;
x = L/N*(0:N-1)';             % x coordinate
y = L/N*(0:N-1)';             % y coordinate
[xx,yy] = meshgrid(x,y);      % xy grid


Ek = zeros(N/2+1,length(t));

qlmodes = zeros(4,nt); % save 1,0; 0,1; 1,1; -1,1 modes in order
% qlmodesa = zeros(4,nt); % save 1,0; 0,1; 1,1; -1,1 modes in order
reversestr = '';
for kk = 1:nt
    if ~mod(kk,100)
       percentdone = 100 * kk / nt;
       msg = sprintf('Percent done: %3.1f', percentdone);
       fprintf(strcat(reversestr, msg));
       reversestr = repmat(sprintf('\b'), 1, length(msg));
    end
    
    qq = fread(fidqh,2*Ntotal,'double');
    qq = complex(qq(1:Ntotal), qq(Ntotal+1:2*Ntotal));

    qh = reshape(qq,N,N)/(N)^2;
    psi_hat = InvBT.*qh;
    vh = Knorm.*psi_hat;
    
    qlmodes(1,kk) = vh(1,2); % 1,0
    qlmodes(2,kk) = vh(2,1); % 0,1
    qlmodes(3,kk) = vh(2,2); % 1,1
    qlmodes(4,kk) = vh(2,N); % -1,1
    
    % conjugate modes (not needed)
%     qlmodesa(1,kk) = qh(1,N); % 1,0
%     qlmodesa(2,kk) = qh(N,1); % 0,1
%     qlmodesa(3,kk) = qh(N,N); % 1,1
%     qlmodesa(4,kk) = qh(N,2); % -1,1

end
%%
handle = figure(1);
clf;
labels = ["(1,0)", "(0,1)", "(1,1)", "(-1,1)"];
for i = 1:size(qlmodes,1)
    subplot(4,1,i)
    [Rmm,lags] = xcorr((qlmodes(i,:).'- mean(qlmodes(i,:),2).'),'coeff');
    Rmm = Rmm(lags>0);
    lags = lags(lags>0);
    plot(lags*dt,real(Rmm)); hold on;
%     plot(lags*dt,imag(Rmm))
    if i == 1 || i == 2
    xlim([0 5e3*dt])
    else
    xlim([0 1e4*dt]); 
    end
    title(sprintf('mode %s',labels(i)));
    xlabel('t');
end
set(gcf,'Position',[867 204 348 709]);


if fsavefigs
    % resizefonts(handle);
    export_fig(fullfile(figfolder, sprintf('autocorr')), '-q101','-p.01','-m1','-pdf');
    savefig(handle, fullfile(figfolder, sprintf('autocorr.fig')))
end

