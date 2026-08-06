% export_snapshots.m
function export_snapshots(dtsave)
if nargin < 1
    dtsave = 0.01;  
end

% Point explicitly to the matlab data folder
manifest_path = fullfile('data','vivid_ensemble_manifest.mat');
if ~exist(manifest_path, 'file')
    error('Could not find manifest at %s. Did you run generate_ensemble.m?', manifest_path);
end

load(manifest_path, 'manifest');

for irun = 1:numel(manifest)
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
    
    fprintf('Exporting snapshots for %s (%s)...\n', datafolder, manifest(irun).split);
    
    [qhc, tsave] = coarsenqh(datafolder, dtsave);
    dataext = load(fullfile(datafolder, 'diagnostics.mat'));
    params = dataext.params;
    N = params.N;
    n_tsave = numel(tsave);
    X = zeros(N, N, n_tsave);
    
    for it = 1:n_tsave
        qk = reshape(qhc(:,it), [N, N]);
        X(:,:,it) = real(ifft2(qk));
    end
    
    snapshot_path = fullfile(datafolder, 'state_snapshots.mat');
    save(snapshot_path, 'X', 'tsave', 'params', '-v7.3');
    fprintf('  -> saved %d snapshots to %s\n', n_tsave, snapshot_path);
end
fprintf('Done. All snapshots are ready for the Python pipeline.\n');
end