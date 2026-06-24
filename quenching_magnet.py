"""
Fluorescence quenching vs magnet distance — camera only, pixel-resolved.

Workflow:
  1. Remove magnet → press Enter → script takes a single baseline.
  2. Place magnet at a distance → type that distance (mm) → script takes signal frames.
  3. Move magnet to next distance, repeat step 2.
  4. Type 'q' to stop and show plots.

Output:
  <OUT_DIR>/quenching.npz
    baseline     : (h, w)              mean intensity, no magnet
    distances    : (n_points,)         mm, as entered
    signal_images: (n_points, h, w)    mean intensity per distance
    quenching_images: (n_points, h, w) baseline − signal  (counts)
    quenching_frac  : (n_points, h, w) (baseline − signal) / baseline
    roi_params   : [ROI_X, ROI_Y, ROI_HALF]
    exposure_s, n_frames : scalar arrays

Requires:  pip install pylablib numpy matplotlib
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pylablib.devices import Thorlabs

# ============================== EDIT THIS ==============================

OUT_DIR  = "quenching_data_no_flick"
NPZ_FILE = "quenching.npz"

EXPOSURE_S = 2e-3   # exposure per frame
N_FRAMES   = 300       # frames averaged per measurement step

# ROI centre in camera pixels (find in ThorCam).
ROI_X, ROI_Y = 1750, 1070
ROI_HALF     = 20   # window is (2*ROI_HALF+1) × (2*ROI_HALF+1)

# =======================================================================


FRAME_TIMEOUT = max(EXPOSURE_S * 20 + 2, 10)  # per-frame timeout in seconds


def acquire_mean(cam, n_frames, roi_x, roi_y, roi_half):
    """Grab n_frames, return per-pixel mean and std over the ROI."""
    time.sleep(0.3)
    print(f"    [dbg] trigger_mode={cam.get_trigger_mode()!r}  armed={cam.acquisition_in_progress()}", flush=True)

    frames = cam.grab(n_frames, frame_timeout=FRAME_TIMEOUT)

    if roi_half is None:
        stack = np.stack([img.astype(np.float64) for img in frames])
    else:
        r0, r1 = roi_y - roi_half, roi_y + roi_half + 1
        c0, c1 = roi_x - roi_half, roi_x + roi_half + 1
        stack  = np.stack([img[r0:r1, c0:c1].astype(np.float64) for img in frames])

    return stack.mean(axis=0), stack.std(axis=0, ddof=1)


def save_dataset(path, baseline, distances, signals, quenchings, quenching_fracs):
    np.savez_compressed(
        path,
        baseline         = baseline,
        distances        = np.array(distances, dtype=np.float64),
        signal_images    = np.stack(signals),
        quenching_images = np.stack(quenchings),
        quenching_frac   = np.stack(quenching_fracs),
        roi_params       = np.array([ROI_X, ROI_Y, ROI_HALF if ROI_HALF is not None else -1]),
        exposure_s       = np.array([EXPOSURE_S]),
        n_frames         = np.array([N_FRAMES]),
    )


def show_distance_point(baseline, signal, quenching_frac, dist_mm, idx):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Distance {dist_mm:.1f} mm  (point {idx + 1})", fontsize=12)

    baseline       = np.rot90(baseline)
    signal         = np.rot90(signal)
    quenching_frac = np.rot90(quenching_frac)

    vmin_fl = min(baseline.min(), signal.min())
    vmax_fl = max(baseline.max(), signal.max())

    im0 = axes[0].imshow(baseline, vmin=vmin_fl, vmax=vmax_fl, origin="upper")
    axes[0].set_title("Baseline (no magnet)")
    plt.colorbar(im0, ax=axes[0], label="counts")

    im1 = axes[1].imshow(signal, vmin=vmin_fl, vmax=vmax_fl, origin="upper")
    axes[1].set_title(f"Signal at {dist_mm:.1f} mm")
    plt.colorbar(im1, ax=axes[1], label="counts")

    absmax = np.nanmax(np.abs(quenching_frac))
    im2 = axes[2].imshow(quenching_frac * 100, vmin=0, vmax=absmax * 100,
                         cmap="turbo", origin="upper")
    axes[2].set_title("Quenching  (baseline − signal) / baseline × 100 %")
    plt.colorbar(im2, ax=axes[2], label="%")

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, f"quenching_d{dist_mm:.1f}mm.png")
    plt.savefig(fig_path, dpi=150)
    plt.show(block=False)
    plt.pause(0.5)
    print(f"  Plot saved → {fig_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUT_DIR, NPZ_FILE)

    baseline        = None
    distances       = []
    signals         = []
    quenchings      = []
    quenching_fracs = []

    def _try_load(path):
        """Load distance-sweep dataset; return (baseline, distances, signals, quenchings, fracs) or None."""
        if not os.path.exists(path):
            return None
        d = np.load(path)
        if "baseline" not in d.files:
            return None  # old per-cycle format, skip
        saved_roi = d["roi_params"]
        if not np.array_equal(saved_roi, expected_roi):
            return None
        n = len(d["distances"])
        return (
            d["baseline"],
            list(d["distances"]),
            [d["signal_images"][i]    for i in range(n)],
            [d["quenching_images"][i] for i in range(n)],
            [d["quenching_frac"][i]   for i in range(n)],
        )

    expected_roi = np.array([ROI_X, ROI_Y, ROI_HALF if ROI_HALF is not None else -1])
    prev_path    = os.path.join(OUT_DIR, NPZ_FILE.replace(".npz", "_prev.npz"))

    loaded = _try_load(npz_path) or _try_load(prev_path)
    if loaded:
        baseline, distances, signals, quenchings, quenching_fracs = loaded
        # Always save into the canonical file going forward
        print(f"Resuming — baseline loaded, {len(distances)} distance point(s) already collected.")
    else:
        if os.path.exists(npz_path):
            import shutil, datetime
            stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bak     = npz_path.replace(".npz", f"_bak_{stamp}.npz")
            shutil.copy2(npz_path, bak)
            print(f"NOTE: ROI mismatch — old file backed up to {bak}, starting fresh.")
        else:
            print(f"Starting new dataset → {npz_path}")

    print("Connecting to Thorlabs camera ...")
    cam = Thorlabs.ThorlabsTLCamera()
    cam.set_exposure(EXPOSURE_S)

    trigger_mode = cam.get_trigger_mode()
    print(f"  trigger mode on connect: {trigger_mode!r}")
    if trigger_mode != "int":
        cam.set_trigger_mode("int")
        print(f"  → forced to 'int' (software trigger)")

    time.sleep(1.0)

    h = w = (2 * ROI_HALF + 1) if ROI_HALF is not None else "full chip"
    print(f"  exposure = {EXPOSURE_S * 1e3:.2f} ms,  frames/step = {N_FRAMES}")
    print(f"  ROI: {h} × {w} px centred at ({ROI_X}, {ROI_Y})\n")

    try:
        # --- Baseline (once) ---
        if baseline is None:
            input("Remove magnet, then press Enter to take BASELINE: ")
            print(f"  Acquiring {N_FRAMES} baseline frames ...", end="", flush=True)
            t0 = time.time()
            baseline, baseline_std = acquire_mean(cam, N_FRAMES, ROI_X, ROI_Y, ROI_HALF)
            print(f" done in {time.time() - t0:.1f} s   "
                  f"ROI mean = {baseline.mean():.2f} ± {baseline_std.mean() / np.sqrt(N_FRAMES):.3f}")
            print()
        else:
            print(f"  Using loaded baseline  (ROI mean = {baseline.mean():.2f})\n")

        # --- Distance sweep ---
        while True:
            s = input("Enter magnet distance in mm (or 'q' to finish): ").strip()
            if s.lower() in ("q", "quit", "exit"):
                break
            try:
                dist_mm = float(s)
            except ValueError:
                print("  Not a number, try again.")
                continue

            print(f"  [{dist_mm:.1f} mm] acquiring {N_FRAMES} frames ...", end="", flush=True)
            t0 = time.time()
            signal, signal_std = acquire_mean(cam, N_FRAMES, ROI_X, ROI_Y, ROI_HALF)
            print(f" done in {time.time() - t0:.1f} s   "
                  f"ROI mean = {signal.mean():.2f} ± {signal_std.mean() / np.sqrt(N_FRAMES):.3f}")

            quenching = baseline - signal
            with np.errstate(invalid="ignore", divide="ignore"):
                quenching_frac = np.where(baseline > 0, quenching / baseline, np.nan)

            roi_q_pct = np.nanmean(quenching_frac) * 100
            print(f"  ROI mean quenching: {roi_q_pct:+.2f} %\n")

            distances.append(dist_mm)
            signals.append(signal)
            quenchings.append(quenching)
            quenching_fracs.append(quenching_frac)

            save_dataset(npz_path, baseline, distances, signals, quenchings, quenching_fracs)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.close()

    if not distances:
        print("No distance points collected.")
        return

    n = len(distances)
    print(f"\nSaved {n} distance point(s) to {npz_path}")

    # Sort by distance for display
    order = np.argsort(distances)
    dists_s = [distances[i] for i in order]
    mean_q  = [np.nanmean(quenching_fracs[i]) * 100 for i in order]

    print("\nDistance (mm)  Quenching (%)")
    for d, q in zip(dists_s, mean_q):
        print(f"  {d:10.2f}    {q:+.3f}")

    # Per-point heatmaps
    for i in order:
        show_distance_point(baseline, signals[i], quenching_fracs[i], distances[i], i)

    # Quenching vs distance summary
    plt.figure(figsize=(7, 4))
    plt.plot(dists_s, mean_q, "o-", markersize=6)
    plt.axhline(0, color="gray", ls="--", lw=0.8)
    plt.xlabel("Magnet distance (mm)")
    plt.ylabel("ROI mean quenching (%)")
    plt.title("Fluorescence quenching vs magnet distance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    summary_path = os.path.join(OUT_DIR, "quenching_vs_distance.png")
    plt.savefig(summary_path, dpi=150)
    plt.show()
    print(f"Summary plot saved → {summary_path}")


if __name__ == "__main__":
    main()
