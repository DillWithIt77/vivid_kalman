import matlab.engine
import os
from pathlib import Path

# ==========================================
# GLOBAL CONTROL FLAG
# ==========================================
FORCE_RERUN = True  # Set to True to override checks and rerun all MATLAB steps

print("Starting MATLAB engine...")
eng = matlab.engine.start_matlab('-nodisplay -nosplash')

root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
matlab_code = os.path.join(root_dir, "matlab")
eng.cd(matlab_code)

data_dir = Path(matlab_code) / "data"
manifest_path = data_dir / "vivid_ensemble_manifest.mat"

N = float(64)
dtsave = 0.01

# ----------------------------------------------------
# 1. Check & Run generate_ensemble.m
# ----------------------------------------------------
if manifest_path.exists() and not FORCE_RERUN:
    print(f"Ensemble manifest found at {manifest_path}. Skipping generate_ensemble.m...")
else:
    print("Running generate_ensemble.m...")
    eng.generate_ensemble(N, dtsave, nargout=0)

# ----------------------------------------------------
# 2. Check & Run run_diagnostics.m
# ----------------------------------------------------
FORCE_RERUN = True
# We check if the manifest exists first so we can look inside it, 
# or check if every run folder already contains its 'diagnostics_plot.png'
diagnostics_needed = True
if manifest_path.exists() and not FORCE_RERUN:
    import scipy.io as sio
    mat_data = sio.loadmat(manifest_path, squeeze_me=True)
    manifest = mat_data["manifest"]
    
    # Check if every run folder has its diagnostic plot already generated
    all_plots_exist = True
    for run in manifest:
        folder_name = str(run["datafolder"])
        plot_file = Path(matlab_code) / folder_name / "diagnostics_plot.png"
        if not plot_file.exists():
            all_plots_exist = False
            break
            
    if all_plots_exist:
        diagnostics_needed = False
        print("All run diagnostic plots already exist. Skipping run_diagnostics.m...")

if diagnostics_needed or FORCE_RERUN:
    print(f"Running run_diagnostics.m with dtsave={dtsave}...")
    eng.run_diagnostics(N, dtsave, nargout=0)

# ----------------------------------------------------
# 3. Check & Run export_snapshots.m
# ----------------------------------------------------
# FORCE_RERUN = True
snapshots_needed = True
if manifest_path.exists() and not FORCE_RERUN:
    import scipy.io as sio
    mat_data = sio.loadmat(manifest_path, squeeze_me=True)
    manifest = mat_data["manifest"]
    
    # Check if every run folder already contains 'state_snapshots.mat'
    all_snapshots_exist = True
    for run in manifest:
        folder_name = str(run["datafolder"])
        snapshot_file = Path(matlab_code) / folder_name / "state_snapshots.mat"
        if not snapshot_file.exists():
            all_snapshots_exist = False
            break
            
    if all_snapshots_exist:
        snapshots_needed = False
        print("All snapshot files (state_snapshots.mat) already exist. Skipping export_snapshots.m...")

if snapshots_needed or FORCE_RERUN:
    print(f"Running export_snapshots.m with dtsave={dtsave}...")
    eng.export_snapshots(N, dtsave, nargout=0)

print("Terminating MATLAB engine...")
eng.quit()
print("All MATLAB tasks complete!")