% generate_ensemble.m
clear;
N        = 128;
beta     = 0.05;
dd_fixed = 5E-3;
kf       = 50; epsf = 0; sigf = 1; alpf = 0;   
tdel     = 0;
forcing_type = 'sin';
Nt       = 2e4;   

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
    
    % Let the driver know where to save this specific run's files
    run_folder_name = sprintf('run_%d_tau_%.3g_dd_%.3g', irun, tau0, dd);
    datafolder = fullfile(output_base_dir, run_folder_name);
    
    % Pass or configure your driver to output into this specific datafolder
    % (Assuming Driver_Spectral_ARK4 accepts or handles the directory creation)
    actual_folder = Driver_Spectral_ARK4(N, beta, kf, epsf, sigf, alpf, ...
        dd, tau0, tdel, Nt, forcing_type);
    
    manifest(irun).datafolder = actual_folder; % or datafolder depending on your driver
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