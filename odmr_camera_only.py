"""
CW-ODMR acquisition — camera only, manual MW frequency, pixel-resolved.

Workflow:
  1. Set MW frequency manually on the SG394 front panel.
  2. Run this script.
  3. Type the current frequency (in MHz) at the prompt.
  4. Script takes N_FRAMES frames, averages them pixel-by-pixel, saves to .npz.
  5. Change MW frequency, type new value.  Type 'q' to quit.

Output:
  <OUT_DIR>/odmr_pixels.npz   — pixel-resolved dataset
    freqs    : (n_points,)          MHz
    images   : (n_points, h, w)     mean intensity per pixel per frequency
    roi_params: [ROI_X, ROI_Y, ROI_HALF]
    exposure_s, n_frames            scalar arrays

Use odmr_analyze.py to produce heatmaps and per-pixel spectra.

Requires:  pip install pylablib numpy matplotlib
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pylablib.devices import Thorlabs

# ============================== EDIT THIS ==============================

OUT_DIR  = "baseline_singleframe"
NPZ_FILE = "data.npz"

EXPOSURE_S = 2e-3          # exposure per frame
N_FRAMES   = 1            # frames to average per frequency point

# ROI centre in camera pixels (find in ThorCam).
# ROI_HALF = None uses the whole chip (warning: large files).
ROI_X, ROI_Y = 1500, 900
ROI_HALF     = 50           # window is (2*ROI_HALF+1) × (2*ROI_HALF+1)

MARK_RESONANCE = False

# =======================================================================


def acquire_point(cam, n_frames, roi_x, roi_y, roi_half):
    """Grab n_frames, return per-pixel mean and std over the ROI."""
    t0     = time.time()
    frames = cam.grab(n_frames)
    dt     = time.time() - t0

    if roi_half is None:
        stack = np.stack([img.astype(np.float64) for img in frames])  # (n,H,W)
    else:
        r0, r1 = roi_y - roi_half, roi_y + roi_half + 1
        c0, c1 = roi_x - roi_half, roi_x + roi_half + 1
        stack  = np.stack([img[r0:r1, c0:c1].astype(np.float64) for img in frames])

    mean_img = stack.mean(axis=0)
    std_img  = stack.std(axis=0, ddof=1)
    return mean_img, std_img, dt


def save_dataset(path, freqs, images):
    np.savez_compressed(
        path,
        freqs      = np.array(freqs,  dtype=np.float64),
        images     = np.stack(images),                      # (n, h, w)
        roi_params = np.array([ROI_X, ROI_Y, ROI_HALF if ROI_HALF is not None else -1]),
        exposure_s = np.array([EXPOSURE_S]),
        n_frames   = np.array([N_FRAMES]),
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUT_DIR, NPZ_FILE)

    # resume if file already exists
    freqs  = []
    images = []
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        freqs  = list(d["freqs"])
        images = [d["images"][i] for i in range(len(freqs))]
        print(f"Resuming — loaded {len(freqs)} existing points from {npz_path}")
    else:
        print(f"Starting new dataset → {npz_path}")

    print("Connecting to Thorlabs camera ...")
    cam = Thorlabs.ThorlabsTLCamera()
    cam.set_exposure(EXPOSURE_S)
    h = w = (2 * ROI_HALF + 1) if ROI_HALF is not None else "full chip"
    print(f"  exposure = {EXPOSURE_S*1e3:.2f} ms, frames/point = {N_FRAMES}")
    print(f"  ROI window: {h} × {w} px centred at ({ROI_X}, {ROI_Y})\n")

    try:
        while True:
            s = input("Enter MW frequency in MHz (or 'q' to quit): ").strip()
            if s.lower() in ("q", "quit", "exit"):
                break
            try:
                f_MHz = float(s)
            except ValueError:
                print("  not a number, try again")
                continue

            print(f"  acquiring {N_FRAMES} frames ...", end="", flush=True)
            mean_img, std_img, dt = acquire_point(cam, N_FRAMES, ROI_X, ROI_Y, ROI_HALF)
            print(f" done in {dt:.1f} s")

            roi_mean = mean_img.mean()
            roi_sem  = std_img.mean() / np.sqrt(N_FRAMES)
            print(f"  f = {f_MHz:8.3f} MHz   ROI mean I = {roi_mean:8.2f} ± {roi_sem:.3f}")

            freqs.append(f_MHz)
            images.append(mean_img)
            save_dataset(npz_path, freqs, images)   # incremental save

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.close()

    if not freqs:
        print("No points collected.")
        return

    print(f"\nSaved {len(freqs)} points to {npz_path}")
    print("Run  odmr_analyze.py  to produce heatmaps and pixel spectra.")

    # quick session summary: mean ROI intensity vs frequency
    arr = np.array(sorted(zip(freqs, [img.mean() for img in images])))
    f_s, i_s = arr[:, 0], arr[:, 1]

    plt.figure(figsize=(8, 4))
    plt.plot(f_s, i_s, "o-", markersize=5)
    plt.xlabel("MW frequency (MHz)")
    plt.ylabel("ROI mean intensity (counts)")
    plt.title(f"CW-ODMR session summary ({len(freqs)} pts)")
    if MARK_RESONANCE:
        plt.axvline(2870, color="gray", ls="--", alpha=0.5, label="NV ZFS")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    summary_path = os.path.join(OUT_DIR, "odmr_session_summary.png")
    plt.savefig(summary_path, dpi=150)
    plt.show()
    print(f"Summary plot saved to {summary_path}")


if __name__ == "__main__":
    main()
