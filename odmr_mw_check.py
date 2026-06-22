"""
CW-ODMR acquisition — camera only, MW on/off check.

Workflow:
  1. Run this script.
  2. At each prompt, type 0 (MW off) or 1 (MW on), then ENTER.
  3. Script takes N_FRAMES frames, averages them, appends one line to .dat.
  4. Toggle MW manually on the SG394 between measurements (RF ON/OFF button).
  5. Type 'q' to quit.  Plot shows intensity vs elapsed time, color-coded.

Output:  odmr.dat  columns:  t_s  mw_state  mean_intensity  std_intensity  n_frames

Requires:  pip install pylablib numpy matplotlib
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pylablib.devices import Thorlabs

# ============================== EDIT THIS ==============================

OUT_DIR  = "odmr_run"
DAT_FILE = "odmr.dat"

EXPOSURE_S = 0.030
N_FRAMES   = 300

ROI_X, ROI_Y = 1474, 935
ROI_HALF    = 200               # set to None for whole chip

# =======================================================================


def acquire_point(cam, n_frames, roi):
    ROI_X, ROI_Y, ROI_HALF = roi
    t0 = time.time()
    frames = cam.grab(n_frames)
    dt = time.time() - t0
    means = np.empty(n_frames, dtype=np.float64)
    for k, img in enumerate(frames):
        if ROI_HALF is None:
            means[k] = img.mean()
        else:
            roi_img = img[ROI_Y-ROI_HALF:ROI_Y+ROI_HALF+1,
                          ROI_X-ROI_HALF:ROI_X+ROI_HALF+1]
            means[k] = roi_img.mean()
    return means.mean(), means.std(ddof=1), dt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dat_path = os.path.join(OUT_DIR, DAT_FILE)

    if not os.path.exists(dat_path):
        with open(dat_path, "w") as fh:
            fh.write("# CW-ODMR MW on/off check\n")
            fh.write(f"# exposure_s = {EXPOSURE_S}\n")
            fh.write(f"# n_frames_per_point = {N_FRAMES}\n")
            fh.write(f"# ROI: x={ROI_X}, y={ROI_Y}, half={ROI_HALF}\n")
            fh.write("# columns: t_s  mw_state(0=off,1=on)  mean_intensity  std_intensity  n_frames\n")
        print(f"Created new {dat_path}")
    else:
        print(f"Appending to existing {dat_path}")

    print("Connecting to Thorlabs camera ...")
    cam = Thorlabs.ThorlabsTLCamera()
    cam.set_exposure(EXPOSURE_S)
    print(f"  exposure = {EXPOSURE_S*1000:.2f} ms, frames/point = {N_FRAMES}\n")
    print("At each prompt: 0 = MW off, 1 = MW on, q = quit\n")

    points = []          # (t_s, state, mean, std)
    t_start = None       # set on first measurement

    try:
        while True:
            s = input("MW state (0=off, 1=on, q=quit): ").strip()
            if s.lower() in ("q", "quit", "exit"):
                break
            if s not in ("0", "1"):
                print("  please type 0, 1, or q")
                continue
            state = int(s)

            t_meas_start = time.time()
            if t_start is None:
                t_start = t_meas_start
            t_rel = t_meas_start - t_start

            print(f"  acquiring {N_FRAMES} frames (MW {'ON' if state else 'OFF'}) ...",
                  end="", flush=True)
            m, sd, dt = acquire_point(cam, N_FRAMES, (ROI_X, ROI_Y, ROI_HALF))
            sem = sd / np.sqrt(N_FRAMES)
            print(f" done in {dt:.1f} s")
            print(f"  t = {t_rel:7.1f} s   MW {'ON ' if state else 'OFF'}   "
                  f"I = {m:8.2f} ± {sem:.3f}   (per-frame σ = {sd:.2f})")

            with open(dat_path, "a") as fh:
                fh.write(f"{t_rel:10.3f}  {state:1d}  {m:12.4f}  "
                         f"{sd:10.4f}  {N_FRAMES:5d}\n")
            points.append((t_rel, state, m, sd))

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.close()

    if not points:
        print("No points collected.")
        return

    arr = np.array(points)
    t, state, mean, std = arr[:,0], arr[:,1].astype(int), arr[:,2], arr[:,3]
    sem = std / np.sqrt(N_FRAMES)

    on_mask  = state == 1
    off_mask = state == 0

    plt.figure(figsize=(10, 5))
    # connect all points with a thin gray line to show time order
    plt.plot(t, mean, '-', color='lightgray', linewidth=1, zorder=1)
    # then overplot on/off in distinct colors
    if off_mask.any():
        plt.errorbar(t[off_mask], mean[off_mask], yerr=sem[off_mask],
                     fmt='o', color='C0', markersize=7, capsize=3,
                     label=f"MW OFF (n={off_mask.sum()})", zorder=3)
    if on_mask.any():
        plt.errorbar(t[on_mask], mean[on_mask], yerr=sem[on_mask],
                     fmt='s', color='C3', markersize=7, capsize=3,
                     label=f"MW ON  (n={on_mask.sum()})", zorder=3)

    # horizontal reference lines: mean of each group
    if off_mask.any():
        plt.axhline(mean[off_mask].mean(), color='C0', ls=':', alpha=0.6,
                    label=f"⟨OFF⟩ = {mean[off_mask].mean():.2f}")
    if on_mask.any():
        plt.axhline(mean[on_mask].mean(), color='C3', ls=':', alpha=0.6,
                    label=f"⟨ON⟩  = {mean[on_mask].mean():.2f}")

    plt.xlabel("Time since first measurement (s)")
    plt.ylabel("Mean intensity (counts)")
    plt.title(f"MW on/off check — {N_FRAMES} frames/point, {len(points)} pts")
    plt.grid(alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "mw_check_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\nPlot saved to {plot_path}")

    # quick summary
    if on_mask.any() and off_mask.any():
        d_on  = mean[on_mask].mean()
        d_off = mean[off_mask].mean()
        contrast_pct = (d_off - d_on) / d_off * 100
        print(f"\n⟨I_off⟩ - ⟨I_on⟩ = {d_off - d_on:+.3f} counts  "
              f"({contrast_pct:+.3f}% contrast)")
        # rough significance: combined SEM
        sem_on  = sem[on_mask].mean()  / np.sqrt(on_mask.sum())
        sem_off = sem[off_mask].mean() / np.sqrt(off_mask.sum())
        sigma   = np.hypot(sem_on, sem_off)
        print(f"Combined SEM ≈ {sigma:.3f} counts  "
              f"→ effect is {(d_off-d_on)/sigma:+.1f} σ "
              f"(>3σ = probably real, <2σ = no significant signal)")


if __name__ == "__main__":
    main()
