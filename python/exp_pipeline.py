"""
exp_pipeline.py

End-to-end example wiring the pipeline together on data exported by
matlab/export_snapshots.m. Includes caching checks to skip completed steps 
unless FORCE_RERUN = True.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import scipy.io as sio

from pod import load_state_snapshots, compute_pod_basis
from sensors import build_tessellated_observation
from background_error import make_background, StationaryCovariance, matern32
from vcnn import VCNN, train_vcnn, estimate_P
from vivid_da import da_3dvar, da_vivid, r_rmse, ssim

rng = np.random.default_rng(0)

# ==========================================
# DEVICE SETUP
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# GLOBAL CONTROL FLAG
# ==========================================
FORCE_RERUN = True  # Set to True to override caches and run everything fresh

# Setup directories
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent  # still used below to locate the sibling matlab/ folder
cache_dir = script_dir / "cache_outputs"
cache_dir.mkdir(exist_ok=True)
output_dir = script_dir / "plot_outputs"
output_dir.mkdir(exist_ok=True)
model_output_dir = script_dir / "model_outputs"
model_output_dir.mkdir(exist_ok=True)

# ---- 1. Load data (with caching) ------------------------------------------
processed_data_path = cache_dir / "processed_train_test_data.npz"

if processed_data_path.exists() and not FORCE_RERUN:
    print('Loading cached training and test datasets...')
    cached_data = np.load(processed_data_path, allow_pickle=True)
    X_train = cached_data["X_train"]
    X_test = cached_data["X_test"]
    N = int(cached_data["N"])
    print(f"Loaded from cache: {len(X_train)} training snapshots, {len(X_test)} test snapshots.")
else:
    print('Loading data from MATLAB manifests...')
    target_dtsave = 0.01  

    manifest_path = base_dir / "matlab" / "data" / "vivid_ensemble_manifest.mat"
    if not manifest_path.exists():
        manifest_path = base_dir / "data" / "vivid_ensemble_manifest.mat"

    mat_data = sio.loadmat(manifest_path, squeeze_me=True)
    manifest = mat_data["manifest"]

    train_runs = []
    test_run = None

    for run in manifest:
        folder_name = str(run["datafolder"])
        split = str(run["split"])

        run_path = base_dir / "matlab" / folder_name / "state_snapshots.mat"
        if not run_path.exists():
            run_path = base_dir / folder_name / "state_snapshots.mat"

        if split.lower() in ["train", "training"]:
            train_runs.append(str(run_path))
        elif split.lower() in ["test", "heldout"]:
            test_run = str(run_path)

    TRAIN_RUNS = train_runs
    TEST_RUN = test_run

    X_train_list, t_train_list = [], []
    for p in TRAIN_RUNS:
        X, t = load_state_snapshots(p)
        X_train_list.append(X)
        t_train_list.append(t)
    X_train = np.concatenate(X_train_list, axis=0)   
    X_test, t_test = load_state_snapshots(TEST_RUN)   
    N = X_train.shape[1]

    # Save to cache
    np.savez(processed_data_path, X_train=X_train, X_test=X_test, N=N)
    print(f"Saved processed datasets to cache: {processed_data_path}")

# ---- 2. Build tessellated-observation / target training pairs for VCNN ---
print('Building tessellated observations...')
N_GRID = 10        
OBS_NOISE_STD = 0.0
USE_NONLINEAR_OBS = True

def build_dataset(X, n_grid):
    Ys, Xs = [], []
    for x in X:
        Y_tilde, coords, y = build_tessellated_observation(
            x, n_grid=n_grid, r_s=3, obs_noise_std=OBS_NOISE_STD, nonlinear=USE_NONLINEAR_OBS, rng=rng)
        Ys.append(Y_tilde[None])   
        Xs.append(x[None])
    Ys = torch.tensor(np.stack(Ys), dtype=torch.float32)
    Xs = torch.tensor(np.stack(Xs), dtype=torch.float32)
    return Ys, Xs

Y_train, Xt_train = build_dataset(X_train, N_GRID)
train_loader = DataLoader(TensorDataset(Y_train, Xt_train), batch_size=16, shuffle=True)

n_val = max(1, len(Y_train) // 10)
val_loader = DataLoader(TensorDataset(Y_train[:n_val], Xt_train[:n_val]), batch_size=16)

# ---- 3. Train VCNN (with weight caching & validation tracking) -------------
model = VCNN(channels=48, n_layers=6)
model.to(device)
best_model_path = model_output_dir / "best_vcnn_model.pth"

if best_model_path.exists() and not FORCE_RERUN:
    print(f'Pre-trained VCNN weights found. Loading from {best_model_path}...')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
else:
    print('No pre-trained model found (or FORCE_RERUN is True). Training VCNN...')
    train_vcnn(
        model, 
        train_loader, 
        val_loader=val_loader, 
        n_epochs=20, 
        lr=1e-3, 
        device=device, 
        save_path=best_model_path
    )
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f'Training complete. Best model loaded and ready from {best_model_path}')

# ---- Diagnostic: in-distribution vs out-of-distribution VCNN performance ----
print('Checking VCNN performance on in-distribution (training-run) snapshots...')

def calc_rrmse(x_pred, x_ref):
    return np.linalg.norm(x_pred - x_ref) / np.linalg.norm(x_ref)

n_check = min(10, len(X_train))
check_idx = np.random.default_rng(1).choice(len(X_train), size=n_check, replace=False)

rrmse_v_train = []
for idx in check_idx:
    x_true_train = X_train[idx]
    Y_tilde, coords, y = build_tessellated_observation(
        x_true_train, n_grid=N_GRID, r_s=3, obs_noise_std=OBS_NOISE_STD, rng=rng)
    with torch.no_grad():
        Yt_in = torch.tensor(Y_tilde[None, None], dtype=torch.float32, device=device)
        x_v_train = model(Yt_in).cpu().numpy()[0, 0].astype(np.float64)
    rrmse_v_train.append(calc_rrmse(x_v_train, x_true_train))

rrmse_v_train = np.array(rrmse_v_train)
print(f"In-distribution (train-run) raw VCNN R-RMSE: "
      f"mean={rrmse_v_train.mean():.4f}  std={rrmse_v_train.std():.4f}")

# ---- 4. Estimate P_t from residuals on the validation split (Eq. 19) -----
# Define your correlation scale length (as specified in your paper/settings)
L = 5.0  
residuals_raw = inv_op_residuals(model, val_loader, device=device)  # shape: (n_val, 1, N, N)
# Squeeze out the singleton channel dimension to get (n_val, N, N)
residuals = residuals_raw.squeeze(axis=1)
# Compute the fully empirical, Gaspari-Cohn localized P_t matrix (Eq. 19-21)
P_t_localized = estimate_P(residuals, L=L)

# ---- 5. Run DA comparison on the held-out test run (with caching per snapshot) ----
USE_LOCALIZATION = False
print('Run DA comparison and saving spatial data...')
s_b = 0.02         
r_obs_std = 0.0    

N_TEST_SNAPSHOTS = 20  
n_snapshots = min(N_TEST_SNAPSHOTS, len(X_test))
print(f"Running DA comparison on {n_snapshots} test snapshots "
      f"(out of {len(X_test)} available)...")

results = {"DA": [], "VIVID": []}

for i, x_true in enumerate(X_test[:n_snapshots]):
    snapshot_file = output_dir / f"experiment_snapshot_{i}.npz"

    if snapshot_file.exists() and not FORCE_RERUN:
        snap_data = np.load(snapshot_file, allow_pickle=True)
        x_a_da = snap_data["x_a_da"]
        x_a_vivid = snap_data["x_a_vivid"]
        
        results["DA"].append(r_rmse(x_a_da, x_true))
        results["VIVID"].append(r_rmse(x_a_vivid, x_true))
        print(f"snapshot {i}: Loaded from cache (DA R-RMSE={results['DA'][-1]:.3f}, VIVID R-RMSE={results['VIVID'][-1]:.3f})")
    else:
        x_b, B_cov = make_background(x_true, s_b=s_b, L=L, rng=rng, localize=USE_LOCALIZATION)

        Y_tilde, coords, y_obs = build_tessellated_observation(
            x_true, n_grid=N_GRID, r_s=3, obs_noise_std=r_obs_std, rng=rng)
        coords_t = torch.tensor(coords)

        # Run 3D-Var (DA)
        x_a_da, n_iter_da, j_hist_da = da_3dvar(x_b, y_obs, coords_t, B_cov,
                                                 r_obs_std=max(r_obs_std, 1e-3))

        # Run VIVID
        with torch.no_grad():
            Yt_in = torch.tensor(Y_tilde[None, None], dtype=torch.float32, device=device)
            x_v = model(Yt_in).cpu().numpy()[0, 0].astype(np.float64)
        x_a_vivid, n_iter_vivid, j_hist_vivid = da_vivid(x_b, x_v, y_obs, coords_t,
                                                         B_cov, P_t_localized,
                                                         r_obs_std=max(r_obs_std, 1e-3))

        results["DA"].append(r_rmse(x_a_da, x_true))
        results["VIVID"].append(r_rmse(x_a_vivid, x_true))

        # Save individual snapshot data dictionary
        snapshot_data = {
            "x_true": x_true,
            "x_b": x_b,
            "x_v": x_v,
            "x_a_da": x_a_da,
            "x_a_vivid": x_a_vivid,
            "j_hist_da": np.array(j_hist_da),
            "j_hist_vivid": np.array(j_hist_vivid),
            "coords": coords,
            "y_obs": y_obs
        }
        np.savez(snapshot_file, **snapshot_data)

        print(f"snapshot {i}: Computed & saved | DA R-RMSE={results['DA'][-1]:.3f} "
              f"({n_iter_da} it)  VIVID R-RMSE={results['VIVID'][-1]:.3f} "
              f"({n_iter_vivid} it)")

print(f"\nAll test snapshot spatial data verified successfully in {output_dir}/")
print("\nMean R-RMSE  DA:", np.mean(results["DA"]),
      " VIVID:", np.mean(results["VIVID"]))