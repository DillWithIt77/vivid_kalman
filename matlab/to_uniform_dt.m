function to_uniform_dt(datafolder, dt_uniform)
    time_dataname = fullfile(datafolder, 't.bin');
    qh_dataname = fullfile(datafolder, 'qh.bin');
    
    % 1. Read raw time steps
    fid_t = fopen(time_dataname, 'r');
    t_raw = fread(fid_t, 'double');
    fclose(fid_t);
    
    % 2. Get system dimensions from diagnostics.mat
    dataext = load(fullfile(datafolder, 'diagnostics.mat'));
    Ntruth = dataext.params.N;
    Ntotal = Ntruth * Ntruth;
    
    % 3. Read raw state history file
    fid_q = fopen(qh_dataname, 'r');
    q_data = fread(fid_q, 'double');
    fclose(fid_q);
    
    num_steps = length(t_raw);
    
    % Reshape raw data into [num_steps, 2 * Ntotal]
    q_matrix = reshape(q_data, [2 * Ntotal, num_steps])';
    clear q_data; 
    
    % 4. Define target uniform time grid
    t_uniform = t_raw(1) : dt_uniform : t_raw(end);
    n_uniform = length(t_uniform);
    
    % 5. Process in chunks and save chunks directly to a temporary MAT-file 
    % or preallocate small chunks without ever touching a 500GB allocation.
    uniform_data_path = fullfile(datafolder, sprintf('uniformqh_dt%.2g.mat', dt_uniform));
    
    % Use matfile for incremental writing to disk
    if exist(uniform_data_path, 'file'), delete(uniform_data_path); end
    m = matfile(uniform_data_path, 'Writable', true);
    m.tsave = t_uniform(:);
    
    chunk_size = 4000;
    total_cols = 2 * Ntotal;
    
    for i = 1 : chunk_size : total_cols
        idx = i : min(i + chunk_size - 1, total_cols);
        
        % Interpolate only this small chunk (size: [n_uniform, length(idx)])
        sub_interp = interp1(t_raw, q_matrix(:, idx), t_uniform', 'pchip');
        
        % Separate real and imaginary for this chunk and map to qhc format
        for j = 1:length(idx)
            global_col = idx(j);
            interpolated_signal = sub_interp(:, j);
            
            if global_col <= Ntotal
                % It's part of the real components
                % We will write/update parts of qhc incrementally
                m.q_real(global_col, :) = interpolated_signal';
            else
                % It's part of the imaginary components
                imag_col = global_col - Ntotal;
                m.q_imag(imag_col, :) = interpolated_signal';
            end
        end
    end
    clear q_matrix;
    
    % 6. Combine real and imaginary parts into final complex matrix qhc on disk
    % MATLAB matfile handles variable mapping smoothly
    fprintf('Successfully streamed uniform dataset cache to: %s\n', uniform_data_path);
end