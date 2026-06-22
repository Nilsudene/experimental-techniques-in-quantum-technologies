"""
Camera noise sweep — no MW signal generator required.

Takes N_MEASUREMENTS repeated acquisitions (each N_FRAMES frames) and saves
all raw frames for noise/statistics analysis.  The "frequency" axis is just
the measurement index (0, 1, ..., N_MEASUREMENTS-1).

Output files in OUT_DIR:
  odmr_pixels.npz       — per-pixel means, compatible with odmr_analyze.py
                           freqs       (N_MEASUREMENTS,)  = measurement index
                           images      (N_MEASUREMENTS, h, w)   mean per pixel
                           images_std  (N_MEASUREMENTS, h, w)   std per pixel
  odmr_raw_frames.npz   — all raw frames
                           freqs       (N_MEASUREMENTS,)
                           raw_frames  (N_MEASUREMENTS, N_FRAMES, h, w)  uint16

Requires:  pip install pylablib numpy matplotlib
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pylablib.devices import Thorlabs

# ============================== EDIT THIS ==============================

OUT_DIR = "odmr_noise_sweep"

# --- Camera ---
EXPOSURE_S     = 2e-3    # per-frame exposure
N_FRAMES       = 1     # frames per acquisition
N_MEASUREMENTS = 100       # number of repeated acquisitions

ROI_X, ROI_Y = 1500, 900
ROI_HALF     = 50       # half-width; None = full chip (large files!)

# =======================================================================


def acquire_frames(cam, n_frames, roi_x, roi_y, roi_half):
    frames = cam.grab(n_frames)
    if roi_half is None:
        stack = np.stack([f for f in frames])
    else:
        r0, r1 = roi_y - roi_half, roi_y + roi_half + 1
        c0, c1 = roi_x - roi_half, roi_x + roi_half + 1
        stack  = np.stack([f[r0:r1, c0:c1] for f in frames])
    return stack.astype(np.uint16)


def save_pixels_npz(path, indices, images, images_std):
    np.savez_compressed(
        path,
        freqs      = np.array(indices,    dtype=np.float64),
        images     = np.stack(images),
        images_std = np.stack(images_std),
        roi_params = np.array([ROI_X, ROI_Y,
                                ROI_HALF if ROI_HALF is not None else -1]),
        exposure_s = np.array([EXPOSURE_S]),
        n_frames   = np.array([N_FRAMES]),
    )


def save_raw_npz(path, indices, raw_stacks):
    np.savez_compressed(
        path,
        freqs      = np.array(indices, dtype=np.float64),
        raw_frames = np.stack(raw_stacks),
        roi_params = np.array([ROI_X, ROI_Y,
                                ROI_HALF if ROI_HALF is not None else -1]),
        exposure_s = np.array([EXPOSURE_S]),
        n_frames   = np.array([N_FRAMES]),
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    h_px = (2 * ROI_HALF + 1) if ROI_HALF is not None else "full chip"
    raw_mb = (N_MEASUREMENTS * N_FRAMES * (2 * ROI_HALF + 1) ** 2 * 2 / 1e6
              if ROI_HALF is not None else float("inf"))

    print(f"=== Camera noise sweep ===")
    print(f"  {N_MEASUREMENTS} acquisitions  ×  {N_FRAMES} frames  ×  {EXPOSURE_S*1e3:.0f} ms")
    print(f"  ROI: {h_px} × {h_px} px   raw data ~{raw_mb:.1f} MB\n")

    print("Connecting to camera ...")
    cam = Thorlabs.ThorlabsTLCamera()
    cam.set_exposure(EXPOSURE_S)
    print(f"  exposure = {EXPOSURE_S*1e3:.2f} ms\n")

    indices       = []
    images_done   = []
    std_done      = []
    raw_done      = []

    pixels_path = os.path.join(OUT_DIR, "odmr_pixels.npz")
    raw_path    = os.path.join(OUT_DIR, "odmr_raw_frames.npz")

    t_start = time.time()
    try:
        for i in range(N_MEASUREMENTS):
            t0    = time.time()
            stack = acquire_frames(cam, N_FRAMES, ROI_X, ROI_Y, ROI_HALF)
            dt    = time.time() - t0

            mean_img = stack.astype(np.float64).mean(axis=0)
            std_img  = stack.astype(np.float64).std(axis=0, ddof=1)

            indices.append(float(i))
            images_done.append(mean_img)
            std_done.append(std_img)
            raw_done.append(stack)

            save_pixels_npz(pixels_path, indices, images_done, std_done)
            save_raw_npz(raw_path, indices, raw_done)

            elapsed   = time.time() - t_start
            remaining = elapsed / (i + 1) * (N_MEASUREMENTS - i - 1)
            print(f"  [{i+1:2d}/{N_MEASUREMENTS}]  "
                  f"ROI mean = {mean_img.mean():8.2f}  "
                  f"pixel range [{mean_img.min():.1f}, {mean_img.max():.1f}]  "
                  f"acq {dt:.1f} s  ETA {remaining:.0f} s")

    except KeyboardInterrupt:
        print("\nInterrupted — partial data saved.")
    finally:
        cam.close()

    n_done = len(indices)
    if n_done == 0:
        print("No data collected.")
        return

    print(f"\nSaved {n_done} acquisitions.")
    print(f"  {pixels_path}")
    print(f"  {raw_path}")

    image_cube = np.stack(images_done)   # (n_done, h, w)

    # Per-pixel noise map: std across acquisitions
    noise_map = image_cube.std(axis=0, ddof=1)    # (h, w)
    mean_map  = image_cube.mean(axis=0)            # (h, w)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Noise sweep — {n_done} acquisitions, {N_FRAMES} frames each")

    im0 = axes[0].imshow(mean_map, cmap="viridis", origin="upper")
    axes[0].set_title("Mean intensity (counts)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(noise_map, cmap="magma", origin="upper")
    axes[1].set_title("Noise σ across acquisitions (counts)")
    fig.colorbar(im1, ax=axes[1])

    rel_noise = noise_map / np.where(mean_map > 0, mean_map, np.nan)
    im2 = axes[2].imshow(rel_noise * 100, cmap="inferno", origin="upper")
    axes[2].set_title("Relative noise σ/mean (%)")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    summary_path = os.path.join(OUT_DIR, "noise_summary.png")
    plt.savefig(summary_path, dpi=150)
    plt.show()
    print(f"Summary plot → {summary_path}")

    # Also save the per-pixel noise map as its own npz for later use
    noise_path = os.path.join(OUT_DIR, "noise_maps.npz")
    np.savez_compressed(noise_path, mean_map=mean_map, noise_map=noise_map,
                        rel_noise=rel_noise)
    print(f"Noise maps   → {noise_path}")


if __name__ == "__main__":
    main()
