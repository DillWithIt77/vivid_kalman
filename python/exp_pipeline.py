"""
exp_pipeline.py  (streaming / batched version)

End-to-end example wiring the pipeline together on data exported by
matlab/export_snapshots.m. Includes caching checks to skip completed steps
unless FORCE_RERUN = True.

WHY THIS VERSION IS DIFFERENT
------------------------------
The original script loaded every training run into one big X_train array,
then built a second (nearly full-size) tensor dataset from it for the VCNN.
For a large ensemble that's two-plus full copies of the dataset alive in
RAM simultaneously -- easy to blow past even a large --mem allocation.

This version:
  1. Converts each MATLAB run to a per-run float32 .npy file once (only
     ONE run is ever fully in memory at a time during conversion).
  2. Memory-maps those .npy files (mmap_mode='r'). Indexing a memmap does
     NOT load the whole file into RAM -- the OS pages in only the rows you
     actually touch.
  3. Uses a lazy torch Dataset that pulls ONE snapshot at a time from the
     memmap and computes its tessellated observation on demand, right when
     the DataLoader asks for it. Peak RAM becomes roughly
     `batch_size * snapshot_size`, not `n_snapshots * snapshot_size`.
  4. Applies the same memmap trick to the held-out test run used in the DA
     loop, so that's never fully materialized either.

NOTE ON REPRODUCIBILITY: because sampling now happens per-item (and
potentially across DataLoader worker processes), tessellation is seeded
deterministically per-sample-index rather than by feeding a single shared
`rng` through everything sequentially. Exact numeric results will differ
slightly from the original in-memory version, but the statistics should be
equivalent.
"""

import gc
import bisect
import resource
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from pathlib import Path
import scipy.io as sio

from pod import load_state_snapshots, compute_pod_basis
from sensors import build_tessellated_observation
from background_error import make_background, StationaryCovariance, matern32
from vcnn import VCNN, train_vcnn, estimate_P, inv_op_residuals
from vivid_da import da_3dvar, da_vivid, r_rmse, ssim


def log_mem(tag):
    """Print peak resident memory so far (stdlib only). ru_maxrss is KB on
    Linux. flush=True so this survives even if the process gets SIGKILLed
    by the OOM killer right after."""
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[MEM-DEBUG] {tag}: peak RSS so far = {peak_mb:,.1f} MB", flush=True)


log_mem("script start")

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
FORCE_RERUN = False  # Set to True to override caches and run everything fresh

# Setup directories
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
cache_dir = script_dir / "cache_outputs"
cache_dir.mkdir(exist_ok=True)
npy_cache_dir = cache_dir / "npy_runs"
npy_cache_dir.mkdir(exist_ok=True)
output_dir = script_dir / "plot_outputs"
output_dir.mkdir(exist_ok=True)
model_output_dir = script_dir / "model_outputs"
model_output_dir.mkdir(exist_ok=True)

# ---- 1. Convert MATLAB runs to per-run float32 .npy caches -----------------
# Only ONE run's worth of data is ever fully in RAM at a time during this
# conversion step; everything downstream reads via memmap.
print('Resolving MATLAB manifest...')
manifest_path = base_dir / "matlab" / "data" / "vivid_ensemble_manifest.mat"
if not manifest_path.exists():
    manifest_path = base_dir / "data" / "vivid_ensemble_manifest.mat"

mat_data = sio.loadmat(manifest_path, squeeze_me=True)
manifest = mat_data["manifest"]
del mat_data
gc.collect()

train_run_paths = []
test_run_path = None
for run in manifest:
    folder_name = str(run["datafolder"])
    split = str(run["split"])

    run_path = base_dir / "matlab" / folder_name / "state_snapshots.mat"
    if not run_path.exists():
        run_path = base_dir / folder_name / "state_snapshots.mat"

    if split.lower() in ["train", "training"]:
        train_run_paths.append(run_path)
    elif split.lower() in ["test", "heldout"]:
        test_run_path = run_path

del manifest
gc.collect()


def ensure_npy_cache(mat_path, out_path):
    """Convert one .mat run to a float32 .npy file if not already cached.
    Loads exactly one run into RAM, writes it, then drops it -- never holds
    more than one run's worth of data at once."""
    if out_path.exists() and not FORCE_RERUN:
        return
    X, t = load_state_snapshots(str(mat_path))
    X = X.astype(np.float32, copy=False)
    np.save(out_path, X)
    del X, t
    gc.collect()


print(f'Converting {len(train_run_paths)} training run(s) to npy cache (streamed, one at a time)...')
train_npy_paths = []
for p in train_run_paths:
    out = npy_cache_dir / f"{p.parent.name}_train.npy"
    ensure_npy_cache(p, out)
    train_npy_paths.append(out)
    log_mem(f"after converting run {p.parent.name}")

test_npy_path = npy_cache_dir / "test_run.npy"
ensure_npy_cache(test_run_path, test_npy_path)
log_mem("after converting test run")

# Memory-map for lazy, on-demand access -- this does NOT load the arrays
# into RAM, it just opens the files.
train_memmaps = [np.load(p, mmap_mode='r') for p in train_npy_paths]
test_memmap = np.load(test_npy_path, mmap_mode='r')

N = train_memmaps[0].shape[1]
run_lengths = [m.shape[0] for m in train_memmaps]
n_train_total = int(sum(run_lengths))
print(f"Total training snapshots (virtual, on disk): {n_train_total}, N={N}")
print(f"Test snapshots (virtual, on disk): {test_memmap.shape[0]}")
log_mem("after opening memmaps (no bulk data loaded into RAM yet)")

# ---- 2. Lazy dataset: computes the tessellated observation on demand ------
print('Setting up lazy tessellated-observation dataset...')
N_GRID = 10
OBS_NOISE_STD = 0.0
USE_NONLINEAR_OBS = True


class LazyTessellatedDataset(Dataset):
    """Streams training snapshots from disk (memmap) and computes the
    tessellated observation for each one only when requested. The full
    dataset is never materialized in RAM -- only whatever the current
    batch needs.

    deterministic=False (default): each __getitem__ call draws fresh
    randomness (seeded from OS entropy), so the same sample gets DIFFERENT
    tessellation noise/coords every epoch -- more like the original
    behavior where a single shared `rng` kept advancing, and generally
    better for training since it acts as a form of data augmentation.
    Not reproducible run-to-run.

    deterministic=True: uses a fixed seed derived from `seed + idx`, so
    the same sample always gets the same tessellation, every epoch, every
    run. Useful for debugging or exact reproducibility, at the cost of
    epoch-to-epoch variation.
    """

    def __init__(self, memmaps, n_grid, seed=0, deterministic=False):
        self.memmaps = memmaps
        self.n_grid = n_grid
        self.seed = seed
        self.deterministic = deterministic
        self.cum_lengths = np.cumsum([0] + [m.shape[0] for m in memmaps])

    def __len__(self):
        return int(self.cum_lengths[-1])

    def __getitem__(self, idx):
        run_idx = bisect.bisect_right(self.cum_lengths, idx) - 1
        local_idx = idx - self.cum_lengths[run_idx]
        # np.array(...) copies just THIS ONE snapshot out of the memmap
        x = np.array(self.memmaps[run_idx][local_idx], dtype=np.float32)
        if self.deterministic:
            local_rng = np.random.default_rng(self.seed + idx)
        else:
            local_rng = np.random.default_rng()  # fresh OS entropy each call
        Y_tilde, coords, y = build_tessellated_observation(
            x, n_grid=self.n_grid, r_s=3, obs_noise_std=OBS_NOISE_STD,
            nonlinear=USE_NONLINEAR_OBS, rng=local_rng)
        Y_t = torch.from_numpy(Y_tilde[None].astype(np.float32))
        X_t = torch.from_numpy(x[None])
        return Y_t, X_t


full_train_dataset = LazyTessellatedDataset(train_memmaps, N_GRID, seed=0, deterministic=True)
n_val = max(1, len(full_train_dataset) // 10)
val_dataset = Subset(full_train_dataset, list(range(n_val)))

# num_workers>0 lets multiple CPU workers prefetch batches in parallel while
# the GPU/CPU trains -- tune based on how many cores your Slurm allocation
# gives you (e.g. match --cpus-per-task).
train_loader = DataLoader(full_train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)

log_mem("after building lazy DataLoaders (still no bulk data in RAM)")

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

del train_loader
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()
log_mem("after training + freeing train_loader")

# ---- Diagnostic: in-distribution vs out-of-distribution VCNN performance ----
print('Checking VCNN performance on in-distribution (training-run) snapshots...')


def calc_rrmse(x_pred, x_ref):
    return np.linalg.norm(x_pred - x_ref) / np.linalg.norm(x_ref)


n_check = min(10, len(full_train_dataset))
check_idx = np.random.default_rng(1).choice(len(full_train_dataset), size=n_check, replace=False)

rrmse_v_train = []
for idx in check_idx:
    Yt_in, x_true_t = full_train_dataset[int(idx)]  # pulls one snapshot from disk
    x_true_train = x_true_t.numpy()[0].astype(np.float64)
    with torch.no_grad():
        x_v_train = model(Yt_in[None].to(device)).cpu().numpy()[0, 0].astype(np.float64)
    rrmse_v_train.append(calc_rrmse(x_v_train, x_true_train))
    del Yt_in, x_true_t, x_true_train, x_v_train

rrmse_v_train = np.array(rrmse_v_train)
print(f"In-distribution (train-run) raw VCNN R-RMSE: "
      f"mean={rrmse_v_train.mean():.4f}  std={rrmse_v_train.std():.4f}")
log_mem("after diagnostic block")

# ---- 4. Estimate P_t from residuals on the validation split (Eq. 19) -----
L = 5.0
residuals_raw = inv_op_residuals(model, val_loader, device=device)
residuals = residuals_raw.squeeze(axis=1)
P_t_localized = estimate_P(residuals, L=L)  # dense (N*N, N*N), Eq. 19-21

print(f"P_t_localized shape={P_t_localized.shape} "
      f"({P_t_localized.nbytes/1e6:,.1f} MB)", flush=True)

# Regularized pseudo-inverse, built ONCE and reused for every DA snapshot
# below (rebuilding per-snapshot would be wasteful -- it's the same matrix
# each time). rcond controls how aggressively near-zero singular values
# (expected, since n_val is typically << N*N) are treated as zero.
from vivid_da import _dense_P_inv
P_inv_np = _dense_P_inv(P_t_localized, rcond=1e-6)
P_inv_t = torch.tensor(P_inv_np, dtype=torch.float64)

del residuals_raw, residuals, val_loader, P_t_localized, P_inv_np
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()
log_mem("after computing P_t_localized and its regularized inverse")

# ---- 5. Run DA comparison on the held-out test run (with caching per snapshot) ----
USE_LOCALIZATION = False
print('Run DA comparison and saving spatial data...')
s_b = 0.02
r_obs_std = 0.0

N_TEST_SNAPSHOTS = 20
n_snapshots = min(N_TEST_SNAPSHOTS, test_memmap.shape[0])
print(f"Running DA comparison on {n_snapshots} test snapshots "
      f"(out of {test_memmap.shape[0]} available)...")

results = {"DA": [], "VIVID": []}

for i in range(n_snapshots):
    # Pull just this one snapshot from disk -- the full test set is never
    # loaded into RAM.
    x_true = np.array(test_memmap[i], dtype=np.float64)

    snapshot_file = output_dir / f"experiment_snapshot_{i}.npz"

    if snapshot_file.exists() and not FORCE_RERUN:
        snap_data = np.load(snapshot_file, allow_pickle=True)
        x_a_da = snap_data["x_a_da"]
        x_a_vivid = snap_data["x_a_vivid"]

        results["DA"].append(r_rmse(x_a_da, x_true))
        results["VIVID"].append(r_rmse(x_a_vivid, x_true))
        print(f"snapshot {i}: Loaded from cache (DA R-RMSE={results['DA'][-1]:.3f}, VIVID R-RMSE={results['VIVID'][-1]:.3f})")
        del snap_data, x_a_da, x_a_vivid
    else:
        x_b, B_cov = make_background(x_true, s_b=s_b, L=L, rng=rng, localize=USE_LOCALIZATION)

        Y_tilde, coords, y_obs = build_tessellated_observation(
            x_true, n_grid=N_GRID, r_s=3, obs_noise_std=r_obs_std, rng=rng)
        coords_t = torch.tensor(coords)

        x_a_da, n_iter_da, j_hist_da = da_3dvar(x_b, y_obs, coords_t, B_cov,
                                                 r_obs_std=max(r_obs_std, 1e-3))

        with torch.no_grad():
            Yt_in = torch.tensor(Y_tilde[None, None], dtype=torch.float32, device=device)
            x_v = model(Yt_in).cpu().numpy()[0, 0].astype(np.float64)
        x_a_vivid, n_iter_vivid, j_hist_vivid = da_vivid(x_b, x_v, y_obs, coords_t,
                                                         B_cov, P_inv_t,
                                                         r_obs_std=max(r_obs_std, 1e-3))

        results["DA"].append(r_rmse(x_a_da, x_true))
        results["VIVID"].append(r_rmse(x_a_vivid, x_true))

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

        del (x_b, B_cov, Y_tilde, coords, y_obs, coords_t, x_a_da, x_a_vivid,
             x_v, j_hist_da, j_hist_vivid, snapshot_data)

    del x_true
    gc.collect()
    if device == "cuda" and i % 5 == 0:
        torch.cuda.empty_cache()
    if i % 5 == 0:
        log_mem(f"after DA snapshot {i}")

print(f"\nAll test snapshot spatial data verified successfully in {output_dir}/")
print("\nMean R-RMSE  DA:", np.mean(results["DA"]),
      " VIVID:", np.mean(results["VIVID"]))
log_mem("script end")