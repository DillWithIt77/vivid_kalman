function export_snapshots(N, dtsave)
if nargin < 1 || isempty(N)
    N = 64;
end
if nargin < 2 || isempty(dtsave)
    dtsave = 0.05;
end

    manifest_path = fullfile('data', sprintf('N%d', N), 'vivid_ensemble_manifest.mat');
    if ~isfile(manifest_path)
        error('export_snapshots:missingManifest', ...
            'Could not find %s from current directory (%s).', manifest_path, pwd);
    end

    mdata = load(manifest_path, 'manifest');
    manifest = mdata.manifest;
    n_runs = numel(manifest);

    fprintf('Loaded manifest with %d runs. Exporting snapshots with dtsave=%.3g...\n', n_runs, dtsave);

    for irun = 1:n_runs
        datafolder = manifest(irun).datafolder;
        fprintf('=== Run %d/%d: %s ===\n', irun, n_runs, datafolder);

        % 1. Locate the coarse cache file written by the driver
        coarse_path = fullfile(datafolder, sprintf('coarseqh_dt%.2g.mat', dtsave));
        if ~isfile(coarse_path)
            warning('export_snapshots:runFailed', 'Skipping %s -- could not find %s.', ...
                datafolder, coarse_path);
            continue;
        end

        % 2. Get grid size from diagnostics.mat
        diag_path = fullfile(datafolder, 'diagnostics.mat');
        if ~isfile(diag_path)
            warning('export_snapshots:runFailed', 'Skipping %s -- could not find %s (needed to determine N).', ...
                datafolder, diag_path);
            continue;
        end
        dataext = load(diag_path, 'params');
        N = dataext.params.N;
        Ntotal = N * N;

        % 3. Open the coarse cache read-only via matfile (no full load into RAM)
        m_in = matfile(coarse_path);
        tsave = m_in.tsave;
        tsave = tsave(:);
        num_saves = length(tsave);

        % Guard against a run that stopped early (qlim exceeded / NaN) leaving
        % tsave and qhc with mismatched lengths
        info = whos(m_in, 'qhc');
        num_cols = info.size(2);
        if num_cols ~= num_saves
            warning('export_snapshots:sizeMismatch', ...
                '%s: tsave has %d entries but qhc has %d columns; truncating to %d.', ...
                datafolder, num_saves, num_cols, min(num_saves, num_cols));
            num_saves = min(num_saves, num_cols);
            tsave = tsave(1:num_saves);
        end

        % 4. Set up output matfile -- field names match pod.py's load_state_snapshots
        % (expects f["X"] with shape (N,N,n_tsave) in MATLAB -> (n_tsave,N,N) via
        % h5py, and f["tsave"])
        out_path = fullfile(datafolder, 'state_snapshots.mat');
        if exist(out_path, 'file'), delete(out_path); end
        m_out = matfile(out_path, 'Writable', true);
        m_out.tsave = tsave;
        m_out.N = N;

        fprintf('  Exporting %d snapshots (N=%d) -> %s\n', num_saves, N, out_path);

        % 5. Stream through in chunks: pull a block of spectral columns,
        % ifft2 each one back to physical space, write the block out.
        chunk_size = 50; % tune based on available RAM (N^2 * chunk_size * 8 bytes per real block)
        for c0 = 1:chunk_size:num_saves
            c1 = min(c0 + chunk_size - 1, num_saves);
            nthis = c1 - c0 + 1;

            qhc_chunk = m_in.qhc(1:Ntotal, c0:c1); % Ntotal x nthis, complex

            qp_chunk = zeros(N, N, nthis);
            for jj = 1:nthis
                qk = reshape(qhc_chunk(:, jj), [N, N]);
                qp_chunk(:, :, jj) = real(ifft2(qk));
            end

            m_out.X(1:N, 1:N, c0:c1) = qp_chunk;
        end

        fprintf('  Done: %s\n', out_path);
    end

    fprintf('Done exporting snapshots for all runs.\n');
end