import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "plot_outputs"

# Auto-detect every experiment_snapshot_*.npz present, instead of a hardcoded list.
# This stays in sync automatically as you change N_TEST_SNAPSHOTS in
# run_pipeline_example.py.
snapshot_files = sorted(
    output_dir.glob("experiment_snapshot_*.npz"),
    key=lambda p: int(p.stem.split("_")[-1]),
)

if not snapshot_files:
    raise FileNotFoundError(
        f"No experiment_snapshot_*.npz files found in {output_dir}. Run your pipeline first!"
    )

print(f"Found {len(snapshot_files)} snapshot file(s) in {output_dir}")

# Only show the full per-snapshot spatial/error/convergence plots for these
# indices, so you're not flooded with figure windows when you have 20+
# snapshots. Set to None to show detailed plots for every snapshot.
DETAILED_PLOT_INDICES = {0, 1, 2}


def calc_rrmse(x_pred, x_ref):
    return np.linalg.norm(x_pred - x_ref) / np.linalg.norm(x_ref)


# ---------------------------------------------------------------------------
# Pass 1: load everything, compute metrics, optionally show detailed plots
# ---------------------------------------------------------------------------
rrmse_b, rrmse_da, rrmse_vivid, rrmse_v = [], [], [], []
snapshot_indices = []
x_true_samples = []  # keep a few x_true fields around for the correlation-length estimate

for file_path in snapshot_files:
    snapshot_idx = int(file_path.stem.split("_")[-1])
    print(f"\nProcessing data for snapshot {snapshot_idx} from {file_path}...")
    data = np.load(file_path, allow_pickle=True)

    x_true = data["x_true"]
    x_b = data["x_b"]
    x_a_da = data["x_a_da"]
    x_a_vivid = data["x_a_vivid"]
    j_hist_da = data["j_hist_da"]
    j_hist_vivid = data["j_hist_vivid"]
    coords = data["coords"]

    rb = calc_rrmse(x_b, x_true)
    rda = calc_rrmse(x_a_da, x_true)
    rvivid = calc_rrmse(x_a_vivid, x_true)

    print(f"--- Quantitative Metrics (Snapshot {snapshot_idx}) ---")
    print(f"Background R-RMSE : {rb:.4f}")
    print(f"DA Analysis R-RMSE: {rda:.4f}")
    print(f"VIVID R-RMSE      : {rvivid:.4f}")

    # Raw VCNN output, if saved -- isolates "is the network bad" from
    # "is the VIVID cost-function combination bad"
    if "x_v" in data:
        x_v = data["x_v"]
        rv = calc_rrmse(x_v, x_true)
        print(f"Raw VCNN (x_v) R-RMSE: {rv:.4f}")
        rrmse_v.append(rv)
    else:
        rrmse_v.append(np.nan)

    rrmse_b.append(rb)
    rrmse_da.append(rda)
    rrmse_vivid.append(rvivid)
    snapshot_indices.append(snapshot_idx)
    x_true_samples.append(x_true)

    if DETAILED_PLOT_INDICES is None or snapshot_idx in DETAILED_PLOT_INDICES:
        # --- 1. Plot Optimization Convergence (Cost Function J) ---
        plt.figure(figsize=(8, 5))
        plt.plot(j_hist_da, label='3D-Var (DA)', color='blue')
        plt.plot(j_hist_vivid, label='VIVID', color='orange')
        plt.yscale('log')
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Cost Function $J$', fontsize=14)
        plt.title(f'Optimization Convergence (Snapshot {snapshot_idx})', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

        # --- 2. Spatial Field Comparisons (Unified Color Scale) ---
        spatial_fields = [x_true, x_b, x_a_da, x_a_vivid]
        vmin_spatial = min(f.min() for f in spatial_fields)
        vmax_spatial = max(f.max() for f in spatial_fields)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        im0 = axes[0].imshow(x_true, cmap='viridis', vmin=vmin_spatial, vmax=vmax_spatial)
        axes[0].set_title('True State ($x_{true}$)')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].axis('off')

        im1 = axes[1].imshow(x_b, cmap='viridis', vmin=vmin_spatial, vmax=vmax_spatial)
        axes[1].set_title('Background ($x_b$)')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].axis('off')

        im2 = axes[2].imshow(x_a_da, cmap='viridis', vmin=vmin_spatial, vmax=vmax_spatial)
        axes[2].set_title('DA Analysis')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        axes[2].axis('off')

        im3 = axes[3].imshow(x_a_vivid, cmap='viridis', vmin=vmin_spatial, vmax=vmax_spatial)
        axes[3].set_title('VIVID Analysis')
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()

        # --- 3. Signed Error Fields (Including Background & Diverging Colormap) ---
        err_b = x_b - x_true
        err_da = x_a_da - x_true
        err_vivid = x_a_vivid - x_true

        error_fields = [err_b, err_da, err_vivid]
        vmax_err = max(abs(f).max() for f in error_fields)
        vmin_err = -vmax_err

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        errors_info = [
            (err_b, 'Background Error ($x_b - x_{true}$)'),
            (err_da, 'DA Analysis Error'),
            (err_vivid, 'VIVID Analysis Error'),
        ]
        for ax, (err_field, title_str) in zip(axes, errors_info):
            im = ax.imshow(err_field, cmap='RdBu_r', vmin=vmin_err, vmax=vmax_err)
            ax.set_title(title_str, fontsize=12)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

rrmse_b = np.array(rrmse_b)
rrmse_da = np.array(rrmse_da)
rrmse_vivid = np.array(rrmse_vivid)
rrmse_v = np.array(rrmse_v)
snapshot_indices = np.array(snapshot_indices)
have_x_v = not np.all(np.isnan(rrmse_v))

# ---------------------------------------------------------------------------
# Pass 2: aggregate diagnostics across ALL snapshots
# ---------------------------------------------------------------------------
def summarize(name, arr):
    print(f"{name:12s}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")

print("\n" + "=" * 60)
print(f"AGGREGATE R-RMSE OVER {len(snapshot_indices)} SNAPSHOTS")
print("=" * 60)
summarize("Background", rrmse_b)
summarize("DA", rrmse_da)
summarize("VIVID", rrmse_vivid)
if have_x_v:
    summarize("Raw VCNN", rrmse_v)

vivid_wins = rrmse_vivid < rrmse_da
print(f"\nVIVID beats DA on {vivid_wins.sum()}/{len(vivid_wins)} snapshots "
      f"({100 * vivid_wins.mean():.1f}%)")

if have_x_v:
    v_worse_than_b = np.nanmean(rrmse_v > rrmse_b)
    v_close_to_vivid = np.nanmean(np.abs(rrmse_v - rrmse_vivid) < 0.05)
    print(f"\nRaw VCNN worse than background on {100 * v_worse_than_b:.0f}% of snapshots")
    if v_worse_than_b > 0.5:
        print(">> The raw VCNN output itself is unreliable on most snapshots -- "
              "this points at the network (architecture/training/sensor density), "
              "not the VIVID cost-function weighting. Revisit VCNN training "
              "(loss curves, capacity, epochs) before touching P_cov.")
    elif v_close_to_vivid:
        print(">> VIVID's result tracks closely with the raw VCNN output -- "
              "VIVID is trusting the network too much relative to the background/obs. "
              "Consider increasing P_cov's implied uncertainty (or revisiting how "
              "P_std feeds into P_cov) so VIVID weights the background more.")
    else:
        print(">> VIVID performs worse than the raw VCNN output alone -- "
              "the combination step itself may have a bug, not just a tuning issue.")

# --- Distribution comparison: boxplot across all snapshots ---
box_data = [rrmse_b, rrmse_da, rrmse_vivid]
box_labels = ['Background', 'DA', 'VIVID']
if have_x_v:
    box_data.append(rrmse_v)
    box_labels.append('Raw VCNN')

plt.figure(figsize=(6, 5))
try:
    plt.boxplot(box_data, tick_labels=box_labels)
except TypeError:
    plt.boxplot(box_data, labels=box_labels)
plt.ylabel('R-RMSE', fontsize=12)
plt.title(f'R-RMSE Distribution Across {len(snapshot_indices)} Snapshots', fontsize=13)
plt.grid(True, axis='y', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- Per-snapshot paired comparison: is VIVID consistently better, or noisy? ---
plt.figure(figsize=(9, 5))
plt.plot(snapshot_indices, rrmse_da, 'o-', label='DA', color='blue')
plt.plot(snapshot_indices, rrmse_vivid, 'o-', label='VIVID', color='orange')
plt.plot(snapshot_indices, rrmse_b, 'o--', label='Background', color='gray', alpha=0.6)
if have_x_v:
    plt.plot(snapshot_indices, rrmse_v, 'o:', label='Raw VCNN', color='green', alpha=0.8)
plt.xlabel('Snapshot index', fontsize=12)
plt.ylabel('R-RMSE', fontsize=12)
plt.title('R-RMSE per Snapshot (consistency check)', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Pass 3: empirical spatial correlation length of x_true
# ---------------------------------------------------------------------------
def radial_autocorrelation(field):
    N = field.shape[0]
    f = field - field.mean()
    F = np.fft.fft2(f)
    acf = np.fft.ifft2(F * np.conj(F)).real
    acf /= acf[0, 0]
    acf = np.fft.fftshift(acf)

    center = N // 2
    yy, xx = np.indices((N, N))
    r = np.sqrt((xx - center) ** 2 + (yy - center) ** 2).astype(int)

    r_max = N // 2
    radial_mean = np.array([acf[r == rr].mean() for rr in range(r_max)])
    return radial_mean


n_for_acf = min(len(x_true_samples), 10)
acf_stack = np.stack([radial_autocorrelation(x) for x in x_true_samples[:n_for_acf]])
acf_mean = acf_stack.mean(axis=0)

below = np.where(acf_mean < 1 / np.e)[0]
L_est = below[0] if len(below) > 0 else len(acf_mean) - 1

print(f"\nEstimated spatial correlation length (1/e crossing): "
      f"L_est ~= {L_est} grid points (averaged over {n_for_acf} true-state snapshots)")

plt.figure(figsize=(7, 5))
plt.plot(acf_mean, marker='o', markersize=3)
plt.axhline(1 / np.e, color='red', ls='--', label='1/e threshold')
plt.axvline(L_est, color='green', ls='--', label=f'L_est = {L_est}')
plt.xlabel('Radius (grid points)', fontsize=12)
plt.ylabel('Radially-averaged autocorrelation', fontsize=12)
plt.title(f'Empirical Spatial Correlation Length of $x_{{true}}$ '
          f'(n={n_for_acf} snapshots)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# NEW: Labeled Multi-Panel Tessellation Sequence Plot
# ---------------------------------------------------------------------------
print("\nGenerating labeled multi-panel tessellation sequence plot...")

n_tess_panels = min(6, len(snapshot_files))
fig, axes = plt.subplots(1, n_tess_panels, figsize=(3.2 * n_tess_panels, 4.0))

if n_tess_panels == 1:
    axes = [axes]

vmin_tess, vmax_tess = 0.0, 0.0030

for idx, ax in enumerate(axes):
    file_path = snapshot_files[idx]
    snapshot_idx = int(file_path.stem.split("_")[-1])
    
    data = np.load(file_path, allow_pickle=True)
    x_true = data["x_true"]
    coords = data["coords"]
    
    ny, nx = x_true.shape
    yy, xx = np.ogrid[:ny, :nx]
    sensor_y = coords[:, 0]
    sensor_x = coords[:, 1]
    
    # Voronoi/nearest-neighbor tessellation blocks from sensor coordinates
    dists = (xx[None, :, :] - sensor_x[:, None, None])**2 + (yy[None, :, :] - sensor_y[:, None, None])**2
    labels = np.argmin(dists, axis=0)
    
    cell_values = x_true[np.clip(sensor_y, 0, ny-1), np.clip(sensor_x, 0, nx-1)]
    tessellated_field = cell_values[labels]
    
    # Plot the tessellated grid map
    im = ax.imshow(tessellated_field, cmap='viridis', vmin=vmin_tess, vmax=vmax_tess, origin='lower', interpolation='nearest')
    
    # Overlay the red sensor dots
    ax.scatter(sensor_x, sensor_y, color='red', s=10, marker='o', edgecolors='none', alpha=0.9)
    
    # Add clear titles indicating what snapshot each panel represents
    ax.set_title(f"Snapshot {snapshot_idx}", fontsize=11, fontweight='bold')
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Attach individual colorbars matching your reference style
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0.0, 0.0010, 0.0020, 0.0030])
    cbar.ax.tick_params(labelsize=8)

# Add an overall descriptive suptitle to the entire figure window
fig.suptitle("Tessellated True State Observations Across Sequential Test Snapshots", fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
plt.show()