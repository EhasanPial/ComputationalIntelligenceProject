import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import SimpleITK as sitk
from scipy.ndimage import zoom


# ── Synapse 9-class colour map ──────────────────────────────────────────────
# class index : (R, G, B)  – background stays transparent
SYNAPSE_COLORS = {
    0: (0,   0,   0),    # background  – will be masked out
    1: (1.0, 0.0, 0.0),  # aorta       – red
    2: (0.0, 0.5, 1.0),  # gallbladder – blue
    3: (0.0, 1.0, 1.0),  # spleen      – cyan
    4: (1.0, 0.0, 1.0),  # left kidney – magenta
    5: (1.0, 0.0, 1.0),  # right kidney– magenta
    6: (1.0, 0.0, 0.0),  # liver       – red  (adjust as needed)
    7: (1.0, 1.0, 0.0),  # stomach     – yellow
    8: (0.0, 1.0, 0.0),  # pancreas    – green
}

CLASS_NAMES = {
    1: "Aorta",
    2: "Gallbladder",
    3: "Spleen",
    4: "L-Kidney",
    5: "R-Kidney",
    6: "Liver",
    7: "Stomach",
    8: "Pancreas",
}

# Match the reference image colour scheme more closely
ORGAN_COLORS = {
    1: (1.0, 0.0, 0.0),   # red
    2: (0.0, 0.0, 1.0),   # blue
    3: (0.0, 1.0, 1.0),   # cyan
    4: (1.0, 0.0, 1.0),   # magenta
    5: (1.0, 0.0, 1.0),   # magenta (right kidney)
    6: (1.0, 0.0, 1.0),   # magenta (liver – large)
    7: (1.0, 1.0, 0.0),   # yellow
    8: (0.0, 1.0, 0.0),   # green
}


def mask_to_rgb(mask, num_classes=9, alpha=0.6):
    """Convert integer segmentation mask → RGBA overlay."""
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for cls_id, color in ORGAN_COLORS.items():
        if cls_id >= num_classes:
            continue
        region = mask == cls_id
        if region.any():
            overlay[region, 0] = color[0]
            overlay[region, 1] = color[1]
            overlay[region, 2] = color[2]
            overlay[region, 3] = alpha          # transparent where bg
    return overlay


def load_nii(path):
    """Load a .nii.gz file and return numpy array (Z, H, W)."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)   # shape: (Z, H, W)
    return arr


def normalise_image(img_slice):
    """Normalise a single 2-D slice to [0, 1]."""
    lo, hi = img_slice.min(), img_slice.max()
    if hi == lo:
        return np.zeros_like(img_slice, dtype=np.float32)
    return ((img_slice - lo) / (hi - lo)).astype(np.float32)


def pick_slices(volume, n=5):
    """Pick n evenly-spaced slices that contain the most foreground."""
    z = volume.shape[0]
    # score each slice by how many labelled pixels it has
    scores = [(volume[i] > 0).sum() for i in range(z)]
    # only consider middle 80 % to avoid blank end-slices
    margin = int(z * 0.1)
    valid = range(margin, z - margin)
    scored = sorted(valid, key=lambda i: scores[i], reverse=True)
    top = sorted(scored[:max(n * 3, 20)])          # keep ordering
    step = max(1, len(top) // n)
    indices = [top[i * step] for i in range(n)][:n]
    return indices


def visualize_case(img_path, pred_path, gt_path=None,
                   n_slices=5, save_path=None, case_name=""):
    """
    Plot a single case as:
      Row 0 – GT overlay  (skipped if gt_path is None)
      Row 1 – Pred overlay
    Each column is one slice.
    """
    img_vol  = load_nii(img_path)
    pred_vol = load_nii(pred_path)
    has_gt   = gt_path is not None and os.path.exists(gt_path)
    if has_gt:
        gt_vol = load_nii(gt_path)

    # Resize pred/gt to match image dims if needed
    if pred_vol.shape != img_vol.shape:
        zoom_f = [img_vol.shape[i] / pred_vol.shape[i] for i in range(3)]
        pred_vol = zoom(pred_vol, zoom_f, order=0)
    if has_gt and gt_vol.shape != img_vol.shape:
        zoom_f = [img_vol.shape[i] / gt_vol.shape[i] for i in range(3)]
        gt_vol  = zoom(gt_vol, zoom_f, order=0)

    slices  = pick_slices(pred_vol if has_gt else pred_vol, n=n_slices)
    n_rows  = 2 if has_gt else 1
    row_labels = (["Ground Truth", "Prediction"] if has_gt else ["Prediction"])

    fig, axes = plt.subplots(n_rows, n_slices,
                             figsize=(3.5 * n_slices, 3.5 * n_rows),
                             facecolor="black")
    if n_rows == 1:
        axes = axes[np.newaxis]          # keep 2-D indexing

    fig.patch.set_facecolor("black")

    for col, idx in enumerate(slices):
        raw = normalise_image(img_vol[idx])

        row_data = []
        if has_gt:
            row_data.append(gt_vol[idx].astype(np.int32))
        row_data.append(pred_vol[idx].astype(np.int32))

        for row, mask in enumerate(row_data):
            ax = axes[row, col]
            ax.imshow(raw, cmap="gray", interpolation="bilinear")
            overlay = mask_to_rgb(mask)
            ax.imshow(overlay, interpolation="nearest")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_labels[row],
                              color="white", fontsize=11,
                              rotation=90, labelpad=6)

    # ── Legend ───────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=ORGAN_COLORS[i], label=CLASS_NAMES[i])
               for i in range(1, 9)]
    fig.legend(handles=patches, loc="upper center",
               ncol=len(patches), fontsize=9,
               framealpha=0, labelcolor="white",
               bbox_to_anchor=(0.5, 1.02))

    title = f"Case: {case_name}" if case_name else "Segmentation Comparison"
    fig.suptitle(title, color="white", fontsize=13, y=1.05)
    plt.tight_layout(pad=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="black")
        print(f"  Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


def visualize_multi_case_grid(pred_dir, img_dir=None, gt_dir=None,
                               n_cases=3, n_slices=5,
                               save_path="comparison_grid.png"):
    """
    Build one large grid  (rows = cases × 2,  cols = n_slices)
    matching the reference image style.
    """
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*_pred.nii.gz")))
    if not pred_files:
        raise FileNotFoundError(f"No *_pred.nii.gz found in {pred_dir}")

    pred_files = pred_files[:n_cases]
    n_rows = n_cases          # one row per case (pred overlay only shown)
                              # set to n_cases*2 if you also want GT rows

    fig, axes = plt.subplots(n_rows, n_slices,
                             figsize=(3.2 * n_slices, 3.2 * n_rows),
                             facecolor="black")
    if n_rows == 1:
        axes = axes[np.newaxis]
    fig.patch.set_facecolor("black")

    for case_idx, pred_path in enumerate(pred_files):
        base = os.path.basename(pred_path).replace("_pred.nii.gz", "")

        pred_vol = load_nii(pred_path)

        # Try to find matching image & GT
        img_path = (os.path.join(img_dir,  base + "_img.nii.gz")
                    if img_dir else pred_path.replace("_pred.", "_img."))
        gt_path  = (os.path.join(gt_dir,   base + "_gt.nii.gz")
                    if gt_dir  else pred_path.replace("_pred.", "_gt."))

        has_img = os.path.exists(img_path)
        has_gt  = os.path.exists(gt_path)

        if has_img:
            img_vol = load_nii(img_path)
            if pred_vol.shape != img_vol.shape:
                zoom_f = [img_vol.shape[i] / pred_vol.shape[i] for i in range(3)]
                pred_vol = zoom(pred_vol, zoom_f, order=0)
        else:
            img_vol = pred_vol   # fallback: use pred as background

        if has_gt:
            gt_vol = load_nii(gt_path)
            if gt_vol.shape != img_vol.shape:
                zoom_f = [img_vol.shape[i] / gt_vol.shape[i] for i in range(3)]
                gt_vol = zoom(gt_vol, zoom_f, order=0)

        slices = pick_slices(pred_vol, n=n_slices)

        for col, idx in enumerate(slices):
            raw  = normalise_image(img_vol[idx])
            mask = pred_vol[idx].astype(np.int32)

            ax = axes[case_idx, col]
            ax.imshow(raw, cmap="gray", interpolation="bilinear")
            ax.imshow(mask_to_rgb(mask), interpolation="nearest")
            ax.axis("off")

        # Row label on the left
        axes[case_idx, 0].set_ylabel(base, color="white",
                                      fontsize=8, rotation=90, labelpad=4)

    # ── Legend ───────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=ORGAN_COLORS[i], label=CLASS_NAMES[i])
               for i in range(1, 9)]
    fig.legend(handles=patches, loc="upper center", ncol=len(patches),
               fontsize=9, framealpha=0, labelcolor="white",
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(pad=0.2)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"\n✅  Grid saved → {save_path}")
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize TransUNet predictions like the reference grid")
    parser.add_argument("--pred_dir",  required=True,
                        help="Folder containing *_pred.nii.gz files")
    parser.add_argument("--img_dir",   default=None,
                        help="Folder containing *_img.nii.gz  (optional, "
                             "defaults to same as pred_dir)")
    parser.add_argument("--gt_dir",    default=None,
                        help="Folder containing *_gt.nii.gz   (optional)")
    parser.add_argument("--n_cases",   type=int, default=3,
                        help="Number of cases to show as rows")
    parser.add_argument("--n_slices",  type=int, default=5,
                        help="Number of slices per case (columns)")
    parser.add_argument("--out",       default="comparison_grid.png",
                        help="Output PNG path")
    parser.add_argument("--single_case", default=None,
                        help="If set, visualise only this case name "
                             "(shows GT vs Pred rows if GT exists)")
    args = parser.parse_args()

    if args.single_case:
        base = args.single_case
        pred_dir = args.pred_dir
        visualize_case(
            img_path  = os.path.join(pred_dir, base + "_img.nii.gz"),
            pred_path = os.path.join(pred_dir, base + "_pred.nii.gz"),
            gt_path   = os.path.join(pred_dir, base + "_gt.nii.gz"),
            n_slices  = args.n_slices,
            save_path = args.out,
            case_name = base,
        )
    else:
        visualize_multi_case_grid(
            pred_dir  = args.pred_dir,
            img_dir   = args.img_dir or args.pred_dir,
            gt_dir    = args.gt_dir  or args.pred_dir,
            n_cases   = args.n_cases,
            n_slices  = args.n_slices,
            save_path = args.out,
        )