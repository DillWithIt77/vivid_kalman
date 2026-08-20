% generate_ensemble.m
function generate_ensemble(uniform_dt)
    N        = 128;
    beta     = 0.05;
    kf       = 50; epsf = 0; sigf = 1; alpf = 0;   
    tdel     = 0;
    forcing_type = 'sin';
    Nt       = 2e4;   
    spinup_frac = 0.1; % fraction of Nt treated as spin-up and excluded from recording

    PARAM_GRID = [ ...
        0.03, 5e-3; ...
        0.04, 5e-3; ...
        0.03, 8e-3; ...
        0.04, 8e-3; ...
        0.05, 6e-3  ... 
    ];
    n_runs = size(PARAM_GRID,1);

    % Define a dedicated data output directory inside your matlab folder
    output_base_dir = fullfile('data');
    if ~exist(output_base_dir, 'dir')
        mkdir(output_base_dir);
    end

    manifest = struct('datafolder', {}, 'tau0', {}, 'dd', {}, 'split', {});
    for irun = 1:n_runs
        tau0 = PARAM_GRID(irun,1);
        dd   = PARAM_GRID(irun,2);
        fprintf('=== Ensemble run %d/%d: tau0=%.3g, dd=%.3g ===\n', irun, n_runs, tau0, dd);
        
        % Run the new solver with on-the-fly uniform time saving[cite: 2, 3]
        actual_folder = Driver_Spectral_ARK4_w_unif(N, beta, kf, epsf, sigf, alpf, ...
            dd, tau0, tdel, Nt, forcing_type, uniform_dt, spinup_frac);

        manifest(irun).datafolder = actual_folder; 
        manifest(irun).tau0 = tau0;
        manifest(irun).dd = dd;
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