% generate_ensemble.m
function generate_ensemble(N, uniform_dt)
    if nargin < 1 || isempty(N)
        N = 64; % default resolution if not specified
    end
    N = double(N);
    beta     = 0.05;
    kf       = 50; epsf = 0; sigf = 1; alpf = 0;
    tdel     = 0;
    forcing_type = 'sin';
    dd       = 1e-2; 

    t_spinup = 3000;
    n_save   = 10000;
    Nt_max   = 6e6;
    n_diag   = 200;

    PARAM_GRID=[ ...
        0.1;...
        0.15;...
        0.2;...
        0.25;...
        0.22;...
    ];
    n_runs = size(PARAM_GRID,1);

    % Manifest lives alongside the run folders the driver already creates
    % under data/N{N}/ -- the driver decides that path itself from N, so
    % we just need to match it here for the manifest.
    output_base_dir = fullfile('data', sprintf('N%d', N));
    if ~exist(output_base_dir, 'dir')
        mkdir(output_base_dir);
    end

    manifest = struct('datafolder', {}, 'tau0', {}, 'split', {}, 'N', {});
    for irun = 1:n_runs
        tau0 = PARAM_GRID(irun,1);
        fprintf('=== Ensemble run %d/%d: tau0=%.3g, N=%d ===\n', irun, n_runs, tau0, N);

        actual_folder = Driver_Spectral_ARK4_w_unif(N, beta, kf, epsf, sigf, alpf, ...
            dd, tau0, tdel, t_spinup, n_save, forcing_type, uniform_dt, Nt_max, n_diag);

        manifest(irun).datafolder = actual_folder;
        manifest(irun).tau0 = tau0;
        manifest(irun).N = N;
        if irun == n_runs
            manifest(irun).split = 'test';
        else
            manifest(irun).split = 'train';
        end
    end

    manifest_path = fullfile(output_base_dir, 'vivid_ensemble_manifest.mat');
    save(manifest_path, 'manifest', 'PARAM_GRID');
    fprintf('Ensemble complete. Manifest saved to %s\n', manifest_path);
end