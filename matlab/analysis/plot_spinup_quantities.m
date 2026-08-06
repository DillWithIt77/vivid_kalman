clear;

fsavefigs = true;

casename = 'baroARK4_N128_beta0.05_d0.005_nu2.5e-28_forcing-sin_tau0-0.04_Nt10000_set-07-13-26-115315';

datafolder = fullfile('data',casename);
dataext = load(fullfile(datafolder, 'diagnostics.mat'));

figfolder = fullfile('figs', casename);
if fsavefigs && ~isdir(figfolder); mkdir(figfolder); end

%%
handle = figure;

N = dataext.params.N;
Ek = dataext.energy;
t = dataext.T;

etotal = sum(Ek,1);

plot(t,etotal);
xlabel('time');
ylabel('total energy');
xlim([t(1) t(end)]);

if fsavefigs
    % resizefonts(handle);
    % export_fig(fullfile(figfolder, sprintf('totalenergy')), '-q101','-p.01','-m1','-pdf');
    savefig(handle, fullfile(figfolder, sprintf('totalenergy.fig')))
end

%%
handle = figure;clf;

ax1 = subplot(1,4,[1 3]);
pcolor(dataext.T,2*pi*(0:N-1)/N,dataext.utz); shading flat;
xlabel('time')
ylabel('y');

title(sprintf('radially averaged zonal velocity u(y)'))

set(gca,'Layer','Top')
set(gcf,'Position',[411 498 740 280]);


ax2 = subplot(1,4,4);
plot(mean(dataext.utz(:,1:end),2),2*pi*(0:N-1)/N); hold on;
plot([0 0],[0 2*pi*(N-1)/N], 'k--');
ylim([0 ,2*pi*(N-1)/N]);
title('mean');
% xlim([-2 1]);

if fsavefigs
    resizefonts(handle);
    ax2.Position([2 4]) = ax1.Position([2 4]);

    % export_fig(fullfile(figfolder, sprintf('avezonalvelocity')), '-q101','-p.01','-m1','-png');
    savefig(handle, fullfile(figfolder, sprintf('avezonalvelocity.fig')))
end
%%
handle = figure; clf;

k = [0:N/2 -N/2+1:-1]';
[KX, KY] = meshgrid(k,k);
Knorm = hypot(KX,KY);
kp = Knorm(1,2:N/2+1); kp = kp(:); 
e = mean(Ek(:,1:end),2);
e = e(2:end);


if strcmp(dataext.params.forcingtype,'constant')
    kkp = (1:35)';

    alpha = -5/3;
else
        kkp = (1:45)';

    alpha = -3;
end
[~,~,ikp] = intersect(kkp,kp);
amin = fminsearch(@(a) mean(log(abs(a*kkp.^alpha - e(ikp)))), 1);


loglog(kp,e); hold on;
loglog(kkp,amin*kkp.^alpha,'k--');
text(2,amin*kkp(1).^alpha,strcat('k^{',strtrim(rats(alpha)),'}'))

xlabel('wavenumber');
ylabel('energy')

if fsavefigs
    % resizefonts(handle);
    % export_fig(fullfile(figfolder, sprintf('spectrum')), '-q101','-p.01','-m1','-pdf');
    savefig(handle, fullfile(figfolder, sprintf('spectrum.fig')))
end

