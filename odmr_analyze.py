"""
ODMR pixel-resolved analysis.

Usage
-----
# Show heatmaps for every frequency point (raw intensity):
  python odmr_analyze.py heatmaps path/to/odmr_pixels.npz

# Show contrast heatmaps  (I/I_ref - 1), ref = mean of off-resonance points:
  python odmr_analyze.py heatmaps path/to/odmr_pixels.npz --contrast

# Save heatmap figures instead of displaying them:
  python odmr_analyze.py heatmaps path/to/odmr_pixels.npz --save

# ODMR spectrum for the pixel at row=5, col=8 in the ROI window:
  python odmr_analyze.py spectrum path/to/odmr_pixels.npz --row 5 --col 8

# Spectrum for multiple pixels (overlaid):
  python odmr_analyze.py spectrum path/to/odmr_pixels.npz --row 5 3 --col 8 12

# Save spectrum plot:
  python odmr_analyze.py spectrum path/to/odmr_pixels.npz --row 5 --col 8 --save

# Baseline drift correction (linear fit subtracted, applied to both subcommands):
  python odmr_analyze.py heatmaps odmr_pixels.npz --baseline baseline.npz
  python odmr_analyze.py spectrum odmr_pixels.npz --row 10 --col 10 --baseline baseline.npz

  The baseline npz is a separate measurement taken under the same conditions
  (e.g. fixed off-resonance MW or no MW) that captures the heating-induced
  intensity drift over time.  A per-pixel linear trend is fitted to the
  baseline images (in acquisition order) and the slope is subtracted from the
  ODMR images before plotting.  The mean level is preserved.

Row / col are 0-indexed from the top-left corner of the ROI window.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_npz(path):
    d = np.load(path)
    freqs  = d["freqs"]                # (n,)
    roi    = d["roi_params"]
    exposure_s = float(d["exposure_s"][0])
    n_frames   = int(d["n_frames"][0])

    # odmr_raw_frames.npz from odmr_auto_sweep.py stores raw_frames (n, n_frames, h, w);
    # compute means on the fly so the rest of the code stays the same.
    if "raw_frames" in d and "images" not in d:
        raw = d["raw_frames"].astype(np.float64)   # (n, n_frames, h, w)
        images = raw.mean(axis=1)
    else:
        images = d["images"]

    return freqs, images, roi, exposure_s, n_frames


def _sort_by_freq(freqs, images):
    order  = np.argsort(freqs)
    return freqs[order], images[order]


# ---------------------------------------------------------------------------
# Baseline drift correction
# ---------------------------------------------------------------------------

def apply_baseline_correction(images, baseline_path):
    """
    Fit a per-pixel linear trend to baseline_images (in acquisition order)
    and subtract the slope from `images` (also in acquisition order).

    The mean level of each pixel is preserved — only the linear drift is removed.

    images        : (n_odmr, h, w)  — ODMR images in acquisition order
    baseline_path : path to a baseline odmr_pixels.npz taken under the same
                    conditions but without (or far from) MW resonance, to
                    capture the heating-induced intensity drift over time.
    """
    d = np.load(baseline_path)
    base = d["images"].astype(np.float64)   # (n_base, h, w)
    n_base, hb, wb = base.shape
    n_odmr, h, w   = images.shape

    if (hb, wb) != (h, w):
        raise ValueError(
            f"Baseline image shape ({hb}×{wb}) does not match "
            f"ODMR image shape ({h}×{w}).  Check ROI settings."
        )

    # Vectorised per-pixel linear fit over baseline acquisition index
    x_base = np.arange(n_base, dtype=np.float64)
    Y      = base.reshape(n_base, -1)           # (n_base, h*w)
    coeffs = np.polyfit(x_base, Y, 1)           # (2, h*w)
    slopes = coeffs[0].reshape(h, w)            # counts / step

    # Map ODMR acquisition indices onto the baseline index range so the
    # slope is comparable even if the two sweeps have different lengths.
    if n_odmr == 1:
        x_odmr = np.array([x_base.mean()])
    else:
        x_odmr = np.linspace(0, n_base - 1, n_odmr)

    # Subtract slope × (index − mean_index) so the mean level is unchanged
    x_centered  = x_odmr - x_odmr.mean()                    # (n_odmr,)
    correction  = x_centered[:, None, None] * slopes[None]  # (n_odmr, h, w)

    corrected = images - correction
    rms = np.sqrt((correction ** 2).mean())
    print(f"  Baseline correction applied: RMS removed = {rms:.2f} counts  "
          f"(slope range [{slopes.min():.3f}, {slopes.max():.3f}] counts/step)")
    return corrected


# ---------------------------------------------------------------------------
# Contrast normalisation
# ---------------------------------------------------------------------------

def contrast_images(freqs, images, ref_frac=0.2):
    """
    Return contrast maps: (I(f) - I_ref) / I_ref
    I_ref is the mean image of the ref_frac outermost frequency points
    (assumed off-resonance).
    """
    n = len(freqs)
    n_ref = max(1, int(round(n * ref_frac)))
    idx   = np.argsort(freqs)
    # take ref_frac from each end
    ref_idx = np.concatenate([idx[:n_ref//2 + n_ref%2], idx[-(n_ref//2):]])
    I_ref   = images[ref_idx].mean(axis=0)
    I_ref   = np.where(I_ref == 0, 1, I_ref)   # avoid div-by-zero
    return (images - I_ref) / I_ref


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmaps(freqs, images, contrast=False, save=False, out_dir="."):
    freqs, images = _sort_by_freq(freqs, images)
    if contrast:
        data  = contrast_images(freqs, images)
        label = "Contrast  (I/I_ref − 1)"
        cmap  = "RdBu_r"
        vmax  = np.percentile(np.abs(data), 99)
        vkw   = dict(vmin=-vmax, vmax=vmax)
    else:
        data  = images
        label = "Intensity (counts)"
        cmap  = "viridis"
        vmax  = np.percentile(data, 99.5)
        vkw   = dict(vmin=0, vmax=vmax)

    n    = len(freqs)
    ncol = min(n, 5)
    nrow = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol,
                              figsize=(3.5 * ncol, 3.2 * nrow),
                              squeeze=False)
    fig.suptitle("ODMR heatmap progression" + (" — contrast" if contrast else ""),
                 fontsize=13)

    imgs_flat = []
    for i, (f, img) in enumerate(zip(freqs, data)):
        r, c  = divmod(i, ncol)
        ax    = axes[r][c]
        im    = ax.imshow(img, cmap=cmap, origin="upper", **vkw)
        ax.set_title(f"{f:.2f} MHz", fontsize=8)
        ax.axis("off")
        imgs_flat.append(im)

    # hide unused axes
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)

    # shared colorbar
    fig.colorbar(imgs_flat[0], ax=axes, label=label, fraction=0.015, pad=0.02)
    plt.tight_layout()

    if save:
        tag  = "contrast" if contrast else "raw"
        path = os.path.join(out_dir, f"odmr_heatmaps_{tag}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Per-pixel spectrum
# ---------------------------------------------------------------------------

def plot_pixel_spectra(freqs, images, rows, cols,
                       contrast=False, save=False, out_dir="."):
    freqs, images = _sort_by_freq(freqs, images)

    if contrast:
        data  = contrast_images(freqs, images)
        ylabel = "Contrast  (I/I_ref − 1)"
    else:
        data   = images
        ylabel = "Intensity (counts)"

    _, h, w = images.shape
    fig, ax  = plt.subplots(figsize=(8, 5))

    colors = cm.tab10(np.linspace(0, 1, len(rows)))
    for (row, col), color in zip(zip(rows, cols), colors):
        if not (0 <= row < h and 0 <= col < w):
            print(f"  WARNING: pixel ({row}, {col}) outside ROI {h}×{w} — skipped")
            continue
        spectrum = data[:, row, col]
        sem      = images[:, row, col].std() / np.sqrt(len(freqs))   # rough error bar
        ax.plot(freqs, spectrum, "o-", markersize=5, color=color,
                label=f"pixel ({row}, {col})")

    ax.set_xlabel("MW frequency (MHz)")
    ax.set_ylabel(ylabel)
    ax.set_title("CW-ODMR — pixel spectra")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save:
        tag  = "contrast" if contrast else "raw"
        pstr = "_".join(f"r{r}c{c}" for r, c in zip(rows, cols))
        path = os.path.join(out_dir, f"odmr_spectrum_{tag}_{pstr}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="ODMR pixel-resolved analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    _baseline_help = (
        "path to a baseline odmr_pixels.npz (same ROI, no MW resonance) "
        "used to fit and subtract the per-pixel linear heating drift"
    )

    # --- heatmaps ---
    ph = sub.add_parser("heatmaps", help="grid of intensity/contrast maps")
    ph.add_argument("npz", help="path to odmr_pixels.npz")
    ph.add_argument("--contrast", action="store_true",
                    help="show (I/I_ref - 1) instead of raw intensity")
    ph.add_argument("--baseline", metavar="NPZ", help=_baseline_help)
    ph.add_argument("--save", action="store_true", help="save figure(s) to file")

    # --- spectrum ---
    ps = sub.add_parser("spectrum", help="ODMR spectrum for specific pixel(s)")
    ps.add_argument("npz", help="path to odmr_pixels.npz")
    ps.add_argument("--row", type=int, nargs="+", required=True,
                    help="row index(es) in ROI window (0 = top)")
    ps.add_argument("--col", type=int, nargs="+", required=True,
                    help="col index(es) in ROI window (0 = left)")
    ps.add_argument("--contrast", action="store_true")
    ps.add_argument("--baseline", metavar="NPZ", help=_baseline_help)
    ps.add_argument("--save", action="store_true")

    args = p.parse_args()

    freqs, images, roi, exposure_s, n_frames = load_npz(args.npz)
    images = images.astype(np.float64)
    out_dir = os.path.dirname(os.path.abspath(args.npz))

    print(f"Loaded {len(freqs)} frequency points, "
          f"image shape {images.shape[1]}×{images.shape[2]} px")
    print(f"Frequencies: {freqs.min():.2f} – {freqs.max():.2f} MHz")
    print(f"ROI params: centre=({int(roi[0])}, {int(roi[1])}), half={int(roi[2])}")

    # Baseline correction is applied in acquisition order, before freq sorting
    if args.baseline:
        print(f"Baseline file: {args.baseline}")
        images = apply_baseline_correction(images, args.baseline)

    if args.cmd == "heatmaps":
        plot_heatmaps(freqs, images,
                      contrast=args.contrast, save=args.save, out_dir=out_dir)

    elif args.cmd == "spectrum":
        rows = args.row
        cols = args.col
        if len(rows) != len(cols):
            sys.exit("ERROR: --row and --col must have the same number of values")
        plot_pixel_spectra(freqs, images, rows, cols,
                           contrast=args.contrast, save=args.save, out_dir=out_dir)


if __name__ == "__main__":
    main()
