function run_diagnostics(N, dtsave)
if nargin < 1 || isempty(N)
    N = 64;
end
if nargin < 2 || isempty(dtsave)
    dtsave = 0.001;
end

clc;

% --- Configuration ---
manifest_path = fullfile('data', sprintf('N%d', N), 'vivid_ensemble_manifest.mat');
p.beta = 0.05;
p.dd = 1e-2;
% p.N is read per-run below from that run's diagnostics.mat, not hardcoded.

fprintf('Using dtsave = %.4g\n', dtsave);

% 1. Load the ensemble manifest
if ~exist(manifest_path, 'file')
    error('Manifest file not found at %s. Please run generate_ensemble.m first.', manifest_path);
end
load(manifest_path, 'manifest', 'PARAM_GRID');
n_runs = length(manifest);

fprintf('Found %d ensemble runs to process.\n', n_runs);

% 2. Loop through each run in the ensemble
for irun = 1:n_runs
    fprintf('\n-----------------------------------------\n');
    fprintf('Processing Run %d/%d (Split: %s)\n', irun, n_runs, manifest(irun).split);
    fprintf('Parameters -> tau0: %.3g\n', manifest(irun).tau0);

    datafolder = manifest(irun).datafolder;

    if ~exist(datafolder, 'dir')
        alt_folder1 = fullfile('matlab', datafolder);
        alt_folder2 = fullfile('..', 'matlab', datafolder);
        
        if exist(alt_folder1, 'dir')
            datafolder = alt_folder1;
        elseif exist(alt_folder2, 'dir')
            datafolder = alt_folder2;
        else
            error('Could not find run datafolder: %s', manifest(irun).datafolder);
        end
    end
    
    % 3. Check for coarsened data; if missing, run coarsenqh automatically
    coarse_file = fullfile(datafolder, sprintf('coarseqh_dt%.2g.mat', dtsave));
    if ~exist(coarse_file, 'file')
        fprintf('  -> Coarsened file missing. Generating via coarsenqh(%s, %.2g)...\n', datafolder, dtsave);
        coarsenqh(datafolder, dtsave);
    else
        fprintf('  -> Found existing coarsened cache file.\n');
    end
    
    % 4. Load the coarsened data variables (qhc and tsave)
    data = load(coarse_file);
    qhc = data.qhc;      % Size: [Ntotal, n_tsave]
    tsave = data.tsave;  % Time vector
    clear data;          % 'data' still held its own copy of qhc -- free it now
    n_tsave = length(tsave);
    dt = tsave(2) - tsave(1);

    % Also load the driver's own accumulated diagnostics (energy, utz, etc.)
    dataext = load(fullfile(datafolder, 'diagnostics.mat'));
    
    % Preallocate arrays to store time-series diagnostics
    vb_time = zeros(n_tsave, 1);
    
    % Prepare wave numbers and operators for advanced metrics (Modes & Spectra)
    N = dataext.params.N;  % read per-run, rather than assuming a fixed p.N
    p.N = N;                % QG_Diagnostics/Spectrum expect N inside p
    k = [0:N/2 -N/2+1:-1]';
    [KX, KY] = meshgrid(k,k);
    Knorm = hypot(KX,KY);
    Knorminv = 1./Knorm; Knorminv(1,1) = 0;
    
    modes = zeros(4, n_tsave); % To track specific modes: (1,0), (0,1), (1,1), (-1,1)[cite: 5]
    
    % 5. Loop through each saved time step and evaluate diagnostics
    fprintf('  -> Running diagnostics and extracting modes across %d time steps...\n', n_tsave);
    for t_idx = 1:n_tsave
        q_hat = reshape(qhc(:, t_idx), [N, N]);
        
        % A. Meridional Heat Flux & Zonal Velocity
        [vb, ~] = QG_Diagnostics(q_hat, p);
        vb_time(t_idx) = vb;
        
        % B. Streamfunction and Specific Modes Extraction for Autocorrelation
        vh = -Knorminv(:).*q_hat(:)/N^2;
        vh_2d = reshape(vh, [N, N]);
        modes(1, t_idx) = vh_2d(1, 2); % (1,0)
        modes(2, t_idx) = vh_2d(2, 1); % (0,1)
        modes(3, t_idx) = vh_2d(2, 2); % (1,1)
        modes(4, t_idx) = vh_2d(2, N); % (-1,1)[cite: 5]
    end

    % --- ADVANCED PLOTTING & SAVING SECTION ---
    
    % Figure 1: Standard Diagnostics (Heat Flux & Energy Spectrum Snapshot)
    fig1 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 900, 400]);
    
    subplot(1, 2, 1);
    plot(real(tsave), real(vb_time), 'LineWidth', 1.5);
    xlabel('Time'); ylabel('Meridional Heat Flux (vb)');
    title(sprintf('Run %d: Heat Flux (tau0=%.2g)', irun, manifest(irun).tau0));
    grid on;
    
    subplot(1, 2, 2);
    [ENE, ~] = Spectrum(reshape(qhc(:, end), [N, N]), p);
    k_vals = 0:(N/2);
    loglog(real(k_vals), real(ENE), '-o', 'LineWidth', 1.5);
    xlabel('Wavenumber (k)'); ylabel('Energy Spectrum (ENE)');
    title('Final State Energy Spectrum');
    grid on;
    
    print(fig1, fullfile(datafolder, 'diagnostics_plot.png'), '-dpng', '-r150', '-painters');
    close(fig1);

    % Figure 2: Mode Autocorrelation Diagnostics (incorporating script logic)
    fig2 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 400, 700]);
    labels = ["(1,0)", "(0,1)", "(1,1)", "(-1,1)"];
    for i = 1:4
        subplot(4,1,i);
        [Rmm, lags] = xcorr((modes(i,:).' - mean(modes(i,:), 2)), 'coeff');
        Rmm = Rmm(lags > 0);
        lags = lags(lags > 0);
        plot(lags*dt, real(Rmm), 'LineWidth', 1.2);
        grid on;
        title(sprintf('Mode Autocorrelation: %s', labels(i)));
        xlabel('t');
        if i <= 2
            xlim([0, min(5e3*dt, max(lags*dt))]);
        else
            xlim([0, min(1e4*dt, max(lags*dt))]);
        end
    end
    
    print(fig2, fullfile(datafolder, 'mode_autocorr_plot.png'), '-dpng', '-r150', '-painters');
    close(fig2);

    % Figure 3: Spin-up total energy vs. time (from driver's accumulated diagnostics)
    try
        fig3 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 600, 400]);
        etotal = sum(dataext.energy, 1);
        plot(dataext.T, etotal, 'LineWidth', 1.5);
        xlabel('time'); ylabel('total energy');
        xlim([dataext.T(1), dataext.T(end)]);
        title(sprintf('Run %d: Spin-up Total Energy', irun));
        grid on;

        print(fig3, fullfile(datafolder, 'spinup_energy_plot.png'), '-dpng', '-r150', '-painters');
        close(fig3);
    catch ME
        warning('  -> Skipped spin-up energy plot for run %d: %s', irun, ME.message);
    end

    % Figure 4: Zonally-averaged velocity Hovmoller plot + mean profile
    try
        fig4 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 740, 280]);
        ax1 = subplot(1, 4, [1 3]);
        pcolor(dataext.T, 2*pi*(0:N-1)/N, dataext.utz); shading flat;
        xlabel('time'); ylabel('y');
        title(sprintf('Run %d: radially averaged zonal velocity u(y)', irun));
        set(gca, 'Layer', 'Top');

        ax2 = subplot(1, 4, 4);
        plot(mean(dataext.utz, 2), 2*pi*(0:N-1)/N); hold on;
        plot([0 0], [0, 2*pi*(N-1)/N], 'k--');
        ylim([0, 2*pi*(N-1)/N]);
        title('mean');
        ax2.Position([2 4]) = ax1.Position([2 4]);

        print(fig4, fullfile(datafolder, 'zonal_velocity_plot.png'), '-dpng', '-r150', '-painters');
        close(fig4);
    catch ME
        warning('  -> Skipped zonal velocity plot for run %d: %s', irun, ME.message);
    end

    % Figure 5: Time-averaged energy spectrum with power-law fit
    try
        fig5 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 500, 400]);
        kp_fit = Knorm(1, 2:N/2+1); kp_fit = kp_fit(:);
        e_mean = mean(dataext.energy, 2);
        e_mean = e_mean(2:end);

        plot_spectrum_with_fit(kp_fit, e_mean, dataext.params.forcingtype);
        title(sprintf('Run %d: Time-averaged Energy Spectrum', irun));

        print(fig5, fullfile(datafolder, 'spectrum_fit_plot.png'), '-dpng', '-r150', '-painters');
        close(fig5);
    catch ME
        warning('  -> Skipped time-averaged spectrum plot for run %d: %s', irun, ME.message);
    end

    % Figure 6: Mean/variance decomposed spectrum (from coarsened streamfunction data)
    try
        fig6 = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 500, 400]);
        [kp_mv, em, ev] = mean_var_spectrum(qhc, N, KX, KY, Knorminv);

        loglog(kp_mv, em+ev, 'LineWidth', 1.25); hold on;
        loglog(kp_mv, ev);
        loglog(kp_mv, em);
        legend('total', 'variance', 'mean');
        overlay_power_law_fit(kp_mv, ev, dataext.params.forcingtype);
        xlabel('wavenumber'); ylabel('energy');
        title(sprintf('Run %d: Mean/Variance Spectral Decomposition', irun));

        print(fig6, fullfile(datafolder, 'spectra_meanvar_plot.png'), '-dpng', '-r150', '-painters');
        close(fig6);
    catch ME
        warning('  -> Skipped mean/variance spectrum plot for run %d: %s', irun, ME.message);
    end
    
    fprintf('  -> Saved standard diagnostics, mode autocorrelations, and extended plots for Run %d.\n', irun);

    % Explicit cleanup between runs -- qhc and dataext are the largest
    % arrays in this script; don't let them linger while the next run loads.
    clear qhc dataext modes vb_time;
end

fprintf('\n=========================================\n');
fprintf('All ensemble diagnostics and extra plots complete!\n');
end

% =========================================================================
% Local helper functions
% =========================================================================

function [kkp, amin, alpha] = fit_power_law(kp, e, forcingtype)
% Fits e ~ amin*k^alpha over a fixed inertial range, choosing the exponent
% based on forcing type (as in plot_spectra.m / plot_spinup_quantities.m).
%
% The fit range is capped at the highest wavenumber actually resolved by
% the grid (max(kp)). Amplitude is solved via closed-form least-squares in
% log-space rather than fminsearch on mean(log(abs(residual))) -- the
% latter objective is unbounded below (it diverges to -Inf whenever the
% fit passes near a data point), which caused "Maximum number of function
% evaluations exceeded / Current function value: -Inf" especially at low
% resolution where the fit range has fewer points.
    kmax = max(kp);
    if strcmp(forcingtype, 'constant')
        kkp = (1:min(35, kmax))';
        alpha = -5/3;
    else
        kkp = (1:min(45, kmax))';
        alpha = -3;
    end
    [~, ~, ikp] = intersect(kkp, kp);
    e_fit = e(ikp);

    % Only use points with strictly positive energy (log undefined at 0/neg)
    valid = e_fit > 0;
    if ~any(valid)
        amin = 1; % degenerate fallback -- no usable points in fit range
        return;
    end
    amin = exp(mean(log(e_fit(valid)) - alpha*log(kkp(valid))));
end

function overlay_power_law_fit(kp, e, forcingtype)
% Overlays a dashed power-law fit line + label on the current loglog axes.
    [kkp, amin, alpha] = fit_power_law(kp, e, forcingtype);
    loglog(kkp, amin*kkp.^alpha, 'k--');
    text(2, amin*kkp(1).^alpha, strcat('k^{', strtrim(rats(alpha)), '}'));
end

function plot_spectrum_with_fit(kp, e, forcingtype)
% Plots a single energy spectrum on loglog axes with a power-law fit overlay.
    loglog(kp, e, 'LineWidth', 1.25); hold on;
    overlay_power_law_fit(kp, e, forcingtype);
    xlabel('wavenumber'); ylabel('energy');
    grid on;
end

function [kp, em, ev] = mean_var_spectrum(qhc, N, KX, KY, Knorminv)
% Decomposes the time series of spectral streamfunction coefficients into
% radially-binned "mean" and "variance" energy spectra (as in plot_spectra.m).
%
% Memory note: this used to build a full [N*N, n_tsave] complex array
% (psi_hat_qhc) the same size as qhc just to take a mean/var across time --
% doubling peak memory on top of qhc already being resident. Replaced with
% a streaming two-pass computation (mean, then variance) column-by-column,
% so extra memory is O(N*N) instead of O(N*N*n_tsave).
    Knorm = hypot(KX, KY);
    kp = Knorm(1, 1:N/2+1);

    n_tsave = size(qhc, 2);
    Knorminv_col = Knorminv(:);

    % Pass 1: streaming mean
    psi_mean = zeros(N*N, 1);
    for t_idx = 1:n_tsave
        psi_mean = psi_mean + (-Knorminv_col .* qhc(:, t_idx) / N^2);
    end
    psi_mean = psi_mean / n_tsave;

    % Pass 2: streaming sample variance (matches var(psi_hat_qhc, 0, 2))
    psi_var = zeros(N*N, 1);
    for t_idx = 1:n_tsave
        psi_t = -Knorminv_col .* qhc(:, t_idx) / N^2;
        psi_var = psi_var + abs(psi_t - psi_mean).^2;
    end
    psi_var = psi_var / (n_tsave - 1);

    qhcm = abs(psi_mean).^2;
    qhcv = psi_var;

    qhcm = reshape(qhcm, N, N);
    qhcv = reshape(qhcv, N, N);

    ev = zeros(N/2+1, 1);
    em = zeros(N/2+1, 1);
    for jj = 1:N
        for ii = 1:N
            kmag = sqrt(KX(ii,jj)^2 + KY(ii,jj)^2);
            if ceil(kmag) <= N/2
                r = kmag - floor(kmag);
                ev(floor(kmag)+1) = ev(floor(kmag)+1) + (1-r)*abs(qhcv(ii,jj));
                ev(ceil(kmag)+1)  = ev(ceil(kmag)+1)  + r*abs(qhcv(ii,jj));

                em(floor(kmag)+1) = em(floor(kmag)+1) + (1-r)*abs(qhcm(ii,jj));
                em(ceil(kmag)+1)  = em(ceil(kmag)+1)  + r*abs(qhcm(ii,jj));
            end
        end
    end
    ev = 0.5*ev;
    em = 0.5*em;
end