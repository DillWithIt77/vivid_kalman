function run_diagnostics(dtsave)
% run_diagnostics.m
% Loads the vivid ensemble manifest, ensures coarsened data exists
% (running coarsenqh on the fly if needed), and computes diagnostics.
%
% dtsave : Desired time interval for coarsening (default 0.001).
%          Exposed as an input argument so this can be called
%          directly from Python, e.g. via the MATLAB engine:
%
%              eng.run_ensemble_diagnostics(0.001, nargout=0)
%
%          or from the command line:
%
%              matlab -batch "run_ensemble_diagnostics(0.001)"

if nargin < 1
    dtsave = 0.001;   % Desired time interval for coarsening
end

clc;

% --- Configuration ---
manifest_path = fullfile('data', 'vivid_ensemble_manifest.mat');
p.N = 128;           % Spatial resolution grid size
p.beta = 0.05;       % Planetary vorticity gradient

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
    fprintf('Parameters -> tau0: %.3g, dd: %.3g\n', manifest(irun).tau0, manifest(irun).dd);
    
    % Update run-specific damping parameter for RHS calculations
    p.dd = manifest(irun).dd;
    datafolder = manifest(irun).datafolder;

    if ~exist(datafolder, 'dir')
        % Try prepending 'matlab/' if it was omitted
        alt_folder1 = fullfile('matlab', datafolder);
        % Try prepending '../matlab/' if running from python context
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
    n_tsave = length(tsave);
    
    % Preallocate arrays to store time-series diagnostics for this run
    vb_time = zeros(n_tsave, 1);
    utz_mean_final = zeros(p.N, 1);
    
    % 5. Loop through each saved time step and evaluate diagnostics
    fprintf('  -> Running diagnostics across %d time steps...\n', n_tsave);
    for t_idx = 1:n_tsave
        % Reshape column vector back to 2D spectral grid [N, N]
        q_hat = reshape(qhc(:, t_idx), [p.N, p.N]);
        
        % A. Compute Meridional Heat Flux & Zonally-averaged Zonal Velocity[cite: 1]
        [vb, utz] = QG_Diagnostics(q_hat, p);
        vb_time(t_idx) = vb;
        
        % Save the zonal velocity from the final step just as an example
        if t_idx == n_tsave
            utz_mean_final = utz;
        end
        
        % B. Compute Energy and Enstrophy Spectra[cite: 3]
        [ENE, ENS] = Spectrum(q_hat, p);
        
        % C. Compute RHS / Tendency terms if needed[cite: 2]
        RHS = RHS_Spectral(q_hat, p);
    end

    % disp(['Size of tsave: ', num2str(size(tsave))]);
    % disp(['Size of vb_time: ', num2str(size(vb_time))]);
    % disp(['Max/Min vb_time: ', num2str(max(vb_time)), ' / ', num2str(min(vb_time))]);
    % --- PLOTTING & SAVING SECTION ---
    fig = figure('Visible', 'off', 'Color', 'w');
    
    % Plot 1: Meridional heat flux over time
    subplot(1, 2, 1);
    plot(real(tsave), real(vb_time), 'LineWidth', 1.5);
    xlabel('Time');
    ylabel('Meridional Heat Flux (vb)');
    title(sprintf('Run %d: Heat Flux (tau0=%.2g)', irun, manifest(irun).tau0));
    grid on;
    
    % Plot 2: Energy spectrum (final time step)
    subplot(1, 2, 2);
    k_vals = 0:(p.N/2);
    loglog(real(k_vals), real(ENE), '-o', 'LineWidth', 1.5);
    xlabel('Wavenumber (k)');
    ylabel('Energy Spectrum (ENE)');
    title('Final State Energy Spectrum');
    grid on;
    
    % Save the figure into the run's specific data folder using fast print
    plot_filename = fullfile(datafolder, 'diagnostics_plot.png');
    
    % This executes instantly without hanging the Python-MATLAB bridge
    print(fig, plot_filename, '-dpng', '-r150', '-painters');
    
    close(fig); % Close the figure to free up memory
    % Optional: You can save individual run diagnostics here or aggregate them
end

fprintf('\n=========================================\n');
fprintf('All ensemble diagnostics complete!\n');
end