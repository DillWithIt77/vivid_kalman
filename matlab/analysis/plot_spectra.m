casename = 'baroARK4_N128_beta0.05_d0.005_nu2.5e-28_forcing-sin_tau0-0.04_Nt50000_set-07-13-26-121958';
datafolder = 'data';
fsavefigs = 0;
    
datafolder = fullfile('data',char(casename));
figfolder = fullfile('figs', char(casename));
if fsavefigs && ~isdir(figfolder); mkdir(figfolder); end

dataext = load(fullfile(datafolder, 'diagnostics.mat'));

qhc = coarsenqh(datafolder, 0); % load data, second argument is junk

Ntruth = sqrt(size(qhc,1));
k = [0:Ntruth/2 -Ntruth/2+1:-1]';
[KX, KY] = meshgrid(k,k);
Knorm = hypot(KX,KY);
Knorminv = 1./Knorm; Knorminv(1,1) = 0;

kp = Knorm(1,1:Ntruth/2+1);


psi_hat_qhc = -Knorminv(:).*qhc/Ntruth^2;
qhcm = abs(mean(psi_hat_qhc,2)).^2;
qhcv = var(psi_hat_qhc,0,2);

qhcm = reshape(qhcm,Ntruth,Ntruth);
qhcv = reshape(qhcv,Ntruth,Ntruth);

ev = zeros(Ntruth/2+1,1);
em = zeros(Ntruth/2+1,1);
for jj = 1:Ntruth
    for ii = 1:Ntruth
        k = sqrt(KX(ii,jj).^2 + KY(ii,jj).^2);
        if ceil(k) <= Ntruth/2
            r = k-floor(k);
            % plus 1 because of matlab indexing
            ev(floor(k)+1) = ev(floor(k)+1) + (1-r)*abs(qhcv(ii,jj));
            ev(ceil(k)+1) = ev(ceil(k)+1) + r*abs(qhcv(ii,jj));
            
            em(floor(k)+1) = em(floor(k)+1) + (1-r)*abs(qhcm(ii,jj));
            em(ceil(k)+1) = em(ceil(k)+1) + r*abs(qhcm(ii,jj));
        end
    end
end
ev = 0.5*ev;
em = 0.5*em;


%%
handle = figure;
loglog(kp,em+ev,'LineWidth',1.25);
hold on;

loglog(kp,ev); 
loglog(kp,em);
legend('total','variance','mean');


if strcmp(dataext.params.forcingtype,'constant')
    kkp = (1:35)';

    alpha = -5/3;
else
        kkp = (1:45)';

    alpha = -3;
end
[~,~,ikp] = intersect(kkp,kp);
amin = fminsearch(@(a) mean(log(abs(a*kkp.^alpha - ev(ikp)))), 1);

loglog(kkp,amin*kkp.^alpha,'k--');
text(2,amin*kkp(1).^alpha,strcat('k^{',strtrim(rats(alpha)),'}'))



xlabel('wavenumber');
ylabel('energy')

drawnow; 
if fsavefigs
    resizefonts(handle);
    export_fig(fullfile(figfolder, sprintf('spectrum')), '-q101','-p.01','-m1','-pdf');
    savefig(handle, fullfile(figfolder, sprintf('spectrum.fig')))
end



