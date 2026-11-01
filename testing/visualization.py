"""
CMD command for creating patches + features for visualization

python ./TRIDENT/run_batch_of_slides.py --task all --wsi_dir ./Data/TCGA-BRCA  
--job_dir ./Data/TCGA-BRCA-processed --patch_encoder virchow2 --mag 20 --patch_size 224 --overlap 0

Generates prediction overlay visualizations for Whole Slide Images using trained models.

Usage:
    python visualization.py --features_dir <path> --model_dir <path> --out_dir <path>

Example:
    python Testing/visualization.py \
        --features_dir ../Data/TCGA-BRCA-processed/20x_224px_70px_overlap \
        --wsi_dir ../Data/TCGA-BRCA \
        --out_dir outputs/visualizations/TCGA-BRCA-new \
"""


import glob
import os
import sys
from typing import Optional, List, Tuple, Dict, Union

from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse

from PIL import Image, ImageDraw
from matplotlib import cm
import openslide

from models import LinearProbe, CrossAttentionEnsemble, HIDDEN_DIM


CKPT_DIR = os.path.join("outputs", "Virchow2")
CKPT_PATH = os.path.join(CKPT_DIR, "best_linear_probe.pt")

# Batch size for inference (to avoid OOM on large slides)
INFERENCE_BATCH_SIZE = 1024

# Confidence threshold for uncertain predictions
CONFIDENCE_THRESHOLD = 0.25

# Mapping from feature directory names to model names used in training
FEATURE_DIR_TO_MODEL = {
    "features_hibou_l": "Hibou-L",
    "features_uni_v2": "UNI2",
    "features_virchow2": "Virchow2",
    "features_hoptimus0": "H-optimus-0",
    "features_conch_v1": "CONCH",
}


def _fetch_mpp(img: openslide.OpenSlide, custom_mpp_keys: Optional[List[str]] = None) -> float:
        """
        Retrieve microns per pixel (MPP) from OpenSlide metadata.

        Parameters
        ----------
        custom_mpp_keys : list of str, optional
            Additional metadata keys to check for MPP.

        Returns
        -------
        float
            MPP value in microns per pixel.

        Raises
        ------
        ValueError
            If MPP cannot be determined from metadata.
        """
        mpp_keys = [
            openslide.PROPERTY_NAME_MPP_X,
            'openslide.mirax.MPP',
            'aperio.MPP',
            'hamamatsu.XResolution',
            'openslide.comment',
        ]

        if custom_mpp_keys:
            mpp_keys.extend(custom_mpp_keys)

        for key in mpp_keys:
            if key in img.properties:
                try:
                    mpp_x = float(img.properties[key])
                    return round(mpp_x, 4)
                except ValueError:
                    continue

        x_resolution = img.properties.get('tiff.XResolution')
        unit = img.properties.get('tiff.ResolutionUnit')

        if x_resolution and unit:
            try:
                if unit.lower() == 'centimeter':
                    return round(10000 / float(x_resolution), 4)
                elif unit.upper() == 'INCH':
                    return round(25400 / float(x_resolution), 4)
            except ValueError:
                pass

        raise ValueError(
            f"Unable to extract MPP from slide metadata"
        )


def _fetch_magnification(img: openslide.OpenSlide):
    mpp_x = _fetch_mpp(img)

    if mpp_x is not None:
        if mpp_x < 0.16:
            return 80
        elif mpp_x < 0.2:
            return 60
        elif mpp_x < 0.3:
            return 40
        elif mpp_x < 0.6:
            return 20
        elif mpp_x < 1.2:
            return 10
        elif mpp_x < 2.4:
            return 5
        else:
            raise ValueError(f"Identified mpp is very low: mpp={mpp_x}. Most WSIs are at 20x, 40x magnification.")


def discover_feature_dirs(base_dir: str, expert_models: List[List[str]]) -> List[str]:
    """Discover feature directories matching the expert models.
    
    Args:
        base_dir: Base directory containing features_* subdirectories
        expert_models: List of expert model configurations from checkpoint
        
    Returns:
        List of feature directory paths in the same order as expert_models
    """
    # Find all features_* directories
    pattern = os.path.join(base_dir, "features_*")
    available_dirs = glob.glob(pattern)
    
    # Build reverse mapping: model name -> feature dir
    model_to_dir = {}
    for feat_dir in available_dirs:
        dir_name = os.path.basename(feat_dir)
        if dir_name in FEATURE_DIR_TO_MODEL:
            model_name = FEATURE_DIR_TO_MODEL[dir_name]
            model_to_dir[model_name] = feat_dir
    
    print(f"Found feature directories in {base_dir}:")
    for dir_name, model_name in FEATURE_DIR_TO_MODEL.items():
        feat_path = os.path.join(base_dir, dir_name)
        if os.path.exists(feat_path):
            print(f"  {dir_name} -> {model_name}")
    
    # Match feature dirs to expert models in order
    feature_dirs = []
    for expert_group in expert_models:
        # For single-model experts (most common case)
        if len(expert_group) == 1:
            model_name = expert_group[0]
            if model_name not in model_to_dir:
                raise ValueError(f"No feature directory found for model: {model_name}. "
                               f"Available: {list(model_to_dir.keys())}")
            feature_dirs.append(model_to_dir[model_name])
        else:
            # For concatenated experts (e.g., ["Hibou-L", "UNI2"]), 
            # this would need special handling - for now just use first
            raise NotImplementedError(
                f"Concatenated expert groups not yet supported for visualization: {expert_group}"
            )
    
    return feature_dirs

def load_model(ckpt_path: str) -> Tuple[nn.Module, Dict]:
    """Load a CrossAttentionEnsemble or single-expert model from checkpoint.

    Ensemble checkpoints carry:  cross_attention_layer, hidden_dim, shared_dim
    Single-expert checkpoints carry:  model_type  ("mlp" | "logreg")
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dictionary")

    is_ensemble           = ckpt.get("is_ensemble", False)
    num_classes           = ckpt["num_classes"]
    expert_embedding_dims = ckpt["expert_embedding_dims"]
    id2name               = ckpt.get("id2name", {i: str(i) for i in range(num_classes)})
    expert_models         = ckpt.get("expert_models", [])

    metadata = {
        "id2name": id2name,
        "num_classes": num_classes,
        "is_ensemble": is_ensemble,
        "expert_models": expert_models,
        "expert_embedding_dims": expert_embedding_dims,
    }

    if is_ensemble:
        hidden_dim            = ckpt.get("hidden_dim", HIDDEN_DIM)
        print(f"Loading CrossAttentionEnsemble: {len(expert_embedding_dims)} experts")
        model = CrossAttentionEnsemble(
            num_classes=num_classes,
            expert_embedding_dims=expert_embedding_dims,
            hidden_dim=hidden_dim
        )
        model.load_state_dict(ckpt["model_state"])
    else:
        model_type = ckpt.get("model_type", "mlp")
        emb_dim    = expert_embedding_dims[0]
        print(f"Loading single-expert model (type={model_type}, emb_dim={emb_dim})")
        if model_type == "logreg":
            model = nn.Linear(emb_dim, num_classes)
        else:
            model = LinearProbe(num_classes=num_classes, embedding_dim=emb_dim)
        model.load_state_dict(ckpt["model_state"])
        metadata["model_type"] = model_type

    model = model.to(device).eval()
    return model, metadata


@torch.inference_mode()
def _run_inference(
    model: nn.Module,
    features_list: List[np.ndarray],
    is_ensemble: bool,
    device: torch.device,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched inference over all tiles of one slide.

    Returns:
        preds:  (N,) int64 class indices
        logits: (N, num_classes) float32
    """
    n_tiles = features_list[0].shape[0]
    all_logits: List[np.ndarray] = []
    for start in range(0, n_tiles, batch_size):
        end = min(start + batch_size, n_tiles)
        if is_ensemble:
            batch = [torch.tensor(f[start:end], dtype=torch.float32).to(device)
                     for f in features_list]
        else:
            batch = torch.tensor(features_list[0][start:end], dtype=torch.float32).to(device)
        all_logits.append(model(batch).cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    return logits_np.argmax(axis=1).astype(np.int64), logits_np
    

def find_wsi_for_slide(slide_id: str, wsi_dir: str) -> Optional[str]:
    """Find the WSI file for a given slide ID.
    
    Args:
        slide_id: Slide identifier
        wsi_dir: Directory to search for WSI files
        
    Returns:
        Path to WSI file or None if not found
    """
    for root, dirs, files in os.walk(wsi_dir):
        for f in files:
            if slide_id in f:
                return os.path.join(root, f)
    return None


def read_wsi_thumbnail(wsi_path: str) -> Image.Image:
    # Deprecated: reading region. Use full-slide thumbnail instead.
    # We ignore region_bbox and return a downsampled whole-slide thumbnail.
    max_side = 3000  # max pixels for the longest side of the thumbnail
    slide = openslide.OpenSlide(wsi_path)
    slide_w, slide_h = slide.dimensions
    # choose target size preserving aspect ratio
    scale = min(1.0, max_side / max(slide_w, slide_h))
    target_w = max(1, int(slide_w * scale))
    target_h = max(1, int(slide_h * scale))

    # OpenSlide provides get_thumbnail which internally picks a level
    try:
        thumb = slide.get_thumbnail((target_w, target_h)).convert("RGB")
    except Exception:
        # fallback: read region at level 0 then resize (may be large and slow)
        thumb = slide.read_region((0, 0), 0, (slide_w, slide_h)).convert("RGB")
        thumb = thumb.resize((target_w, target_h), resample=Image.BILINEAR)
    slide.close()
    return thumb


def load_features_for_slide(slide_id: str, feature_dirs: List[str]) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load features from all expert feature directories for a given slide.
    
    Args:
        slide_id: Slide identifier (without extension)
        feature_dirs: List of directories containing features for each expert
        
    Returns:
        features_list: List of feature arrays, one per expert (each N x D_i)
        coords: Coordinate array (N x 2), same across all experts
    """
    features_list = []
    coords = None
    
    for feat_dir in feature_dirs:
        h5_path = os.path.join(feat_dir, f"{slide_id}.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Feature file not found: {h5_path}")
        
        with h5py.File(h5_path, "r") as f:
            feats = f["features"][:]  # (N, D_i)
            coords_i = f["coords"][:]  # (N, 2)
            
            # Verify coordinates match across all experts
            if coords is None:
                coords = coords_i
            else:
                if not np.array_equal(coords, coords_i):
                    raise ValueError(f"Coordinate mismatch for {slide_id} across feature files")
            
            features_list.append(feats)
    
    return features_list, coords


def compute_stride_from_coords(coords: np.ndarray) -> Optional[int]:
    """Infer the patch stride (in level-0 pixels) from patch coordinates.

    Returns None if only one patch exists (stride cannot be determined).
    """
    for axis in [0, 1]:
        unique_vals = np.sort(np.unique(coords[:, axis]))
        if len(unique_vals) > 1:
            diffs = np.diff(unique_vals)
            return int(np.min(diffs[diffs > 0]))
    return None


def aggregate_overlapping_predictions(
    coords: np.ndarray,
    logits: np.ndarray,
    patch_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate predictions for overlapping patches via Gaussian-weighted logits.

    Each patch of size ``patch_size`` is subdivided into
    (patch_size/stride)^2 sub-patches of size ``stride``.  For every
    sub-patch position we compute a Gaussian-weighted average of the raw
    logits from all overlapping patches, where the weight decays with
    distance from the sub-patch centre to each patch's centre
    (sigma = patch_size / 3).

    This preserves small structures (centre-of-patch predictions dominate)
    while smoothing boundaries (border sub-patches blend neighbours).

    Returns
    -------
    sub_coords : (M, 2) int64
        Top-left corner of each stride x stride sub-patch.
    sub_preds : (M,) int64
        Predicted class (argmax of weighted-average logits).
    sub_confidences : (M,) float32
        Softmax confidence of the winning class.
    """
    from collections import defaultdict

    n_sub = round(patch_size / stride)  # sub-patches per side per patch
    num_classes = logits.shape[1]
    sigma = patch_size / 3.0
    half = patch_size / 2.0

    # Map: (sx, sy) -> [patch_idx, ...]
    subpatch_map: Dict[Tuple[int, int], list] = defaultdict(list)

    for i in range(len(coords)):
        x, y = int(coords[i, 0]), int(coords[i, 1])
        for di in range(n_sub):
            for dj in range(n_sub):
                sx = x + di * stride
                sy = y + dj * stride
                subpatch_map[(sx, sy)].append(i)

    n = len(subpatch_map)
    sub_coords = np.empty((n, 2), dtype=np.int64)
    sub_preds = np.empty(n, dtype=np.int64)
    sub_confidences = np.empty(n, dtype=np.float32)

    for idx, ((sx, sy), patch_indices) in enumerate(subpatch_map.items()):
        sub_coords[idx] = [sx, sy]

        weighted_logit = np.zeros(num_classes, dtype=np.float64)
        w_sum = 0.0
        # sub-patch centre
        scx = sx + stride / 2.0
        scy = sy + stride / 2.0
        for i in patch_indices:
            cx = coords[i, 0] + half
            cy = coords[i, 1] + half
            d2 = (scx - cx) ** 2 + (scy - cy) ** 2
            w = np.exp(-d2 / (2.0 * sigma * sigma))
            weighted_logit += w * logits[i]
            w_sum += w
        if w_sum > 0:
            weighted_logit /= w_sum

        pred_class = int(np.argmax(weighted_logit))
        # softmax of weighted-average logits for confidence
        exp_l = np.exp(weighted_logit - weighted_logit.max())
        soft = exp_l / exp_l.sum()

        sub_preds[idx] = pred_class
        sub_confidences[idx] = float(soft[pred_class])

    return sub_coords, sub_preds, sub_confidences


def overlay_predictions_on_wsi(
    slide_id: str,
    model: nn.Module,
    metadata: Dict,
    feature_dirs: List[str],
    out_dir: str,
    wsi_dir: str,
    magnification: Optional[int] = None,
    predictions_only: bool = False,
    no_legend: bool = False,
    pbar: Optional[tqdm] = None,
):
    """Generate prediction overlay and save CSV for a single slide."""
    out_path = os.path.join(out_dir, f"{slide_id}_overlay.png")
    if os.path.exists(out_path):
        print(f"Skipping {slide_id} (already visualized)")
        return

    is_ensemble = metadata["is_ensemble"]
    device      = next(model.parameters()).device

    # Load features (unified path — always returns a list)
    features_list, coords = load_features_for_slide(slide_id, feature_dirs)

    # Run batched inference
    preds, logits_np = _run_inference(model, features_list, is_ensemble, device)

    probs       = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    confidences = probs.max(axis=1)
    
    # WSI thumbnail
    wsi_path = find_wsi_for_slide(slide_id, wsi_dir)
    if wsi_path is None:
        print(f"Warning: could not find original WSI for {slide_id} in {wsi_dir}, skipping overlay")
        return

    # Get slide dimensions
    slide = openslide.OpenSlide(wsi_path)
    slide_w, slide_h = slide.dimensions

    # If user provided magnification via CLI, prefer that
    if magnification is not None:
        magnification = int(magnification)
    else:
        magnification = _fetch_magnification(slide)

    patch_size = (magnification / 20) * 224

    if pbar is not None:
        pbar.set_postfix(slide=slide_id, mag=magnification, patch_size=int(round(patch_size)))

    slide.close()

    # Aggregate overlapping patch predictions via Gaussian-weighted logits
    patch_size_int = int(round(patch_size))
    stride = compute_stride_from_coords(coords)

    if stride is not None and stride < patch_size_int:
        n_sub = round(patch_size_int / stride)
        print(f"  Overlap detected: stride={stride}, patch_size={patch_size_int}, "
              f"sub-patches per patch={n_sub}x{n_sub}")
        agg_coords, preds, confidences = aggregate_overlapping_predictions(
            coords, logits_np, patch_size_int, stride
        )
        xs = agg_coords[:, 0]
        ys = agg_coords[:, 1]
        draw_size = stride
    else:
        xs = coords[:, 0].astype(int)
        ys = coords[:, 1].astype(int)
        draw_size = patch_size_int
    
    # Save predictions, logits, and coordinates to CSV
    id2name = metadata["id2name"]
    predictions_data = {
        'x': xs.tolist(),
        'y': ys.tolist(),
        'prediction_id': preds.tolist(),
        'prediction_name': [id2name.get(int(p), str(p)) if isinstance(id2name, dict) else str(p) for p in preds],
        'confidence': confidences.tolist(),
    }

    # Add logits columns only when patches don't overlap (logits aren't aggregated)
    if stride is None or stride >= patch_size_int:
        num_classes = metadata["num_classes"]
        for i in range(num_classes):
            class_name = id2name.get(i, str(i)) if isinstance(id2name, dict) else str(i)
            predictions_data[f'logit_{class_name}'] = logits_np[:, i].tolist()
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_csv_dir = os.path.join(out_dir, "CSV")
    ensure_dir(predictions_csv_dir)
    predictions_csv_path = os.path.join(predictions_csv_dir, f"{slide_id}_predictions.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)

    if predictions_only:
        return
        
    thumb = read_wsi_thumbnail(wsi_path)
    thumb_w, thumb_h = thumb.width, thumb.height

    overlay = thumb.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    num_classes = metadata["num_classes"]
    cmap = cm.get_cmap("tab20", num_classes)
    id2name = metadata["id2name"]
    
    # Color for uncertain predictions (black)
    uncertain_color = (0, 0, 0)
    
    # Custom color for Epithelial class
    epithelial_color = (255, 255, 0)
    
    # Find Epithelial class index if it exists
    epithelial_idx = None
    for idx, name in id2name.items():
        if isinstance(name, str) and name.lower() == "epithelial":
            epithelial_idx = idx
            break

    # Draw predictions on thumbnail
    for (x, y, p, conf) in zip(xs, ys, preds, confidences):
        tx = int(x / slide_w * thumb_w)
        ty = int(y / slide_h * thumb_h)
        tw = max(1, int(draw_size / slide_w * thumb_w))
        th = max(1, int(draw_size / slide_h * thumb_h))
        
        # Use uncertain color if confidence is below threshold
        if conf < CONFIDENCE_THRESHOLD:
            color = uncertain_color
            alpha = 0
        elif epithelial_idx is not None and p == epithelial_idx:
            # Use custom color for Epithelial class
            color = epithelial_color
            alpha = int(conf * 255)
        else:
            color = tuple((np.array(cmap(int(p))[:3]) * 255).astype(np.uint8).tolist())
            alpha = int(conf * 255)
        
        draw.rectangle([tx, ty, tx + tw, ty + th], fill=(color[0], color[1], color[2], alpha))

    # Draw legend
    if not no_legend:
        legend_items = min(num_classes, 20)
        legend_h = 18
        legend_w = 220
        # Add one extra row for "Uncertain" class
        legend = Image.new("RGBA", (legend_w, (legend_items + 1) * legend_h), (255, 255, 255, 220))
        ld = ImageDraw.Draw(legend)

        # Draw class legend items
        for i in range(legend_items):
            # Use custom color for Epithelial class in legend
            if epithelial_idx is not None and i == epithelial_idx:
                color = epithelial_color
            else:
                color = tuple((np.array(cmap(i)[:3]) * 255).astype(np.uint8).tolist())
            y0 = i * legend_h
            ld.rectangle([4, y0 + 4, 18, y0 + 14], fill=color)
            label_text = id2name.get(i, str(i)) if isinstance(id2name, dict) else str(i)
            ld.text((24, y0 + 2), label_text, fill=(0, 0, 0, 255))

        # Draw "Uncertain" class at the bottom
        y0 = legend_items * legend_h
        ld.rectangle([4, y0 + 4, 18, y0 + 14], fill=(128, 128, 128))
        ld.text((24, y0 + 2), f"Other", fill=(0, 0, 0, 255))

        # Paste legend onto overlay
        overlay.paste(legend, (6, 6), legend)

    ensure_dir(out_dir)
    composite = Image.alpha_composite(thumb.convert("RGBA"), overlay)
    composite.save(out_path)

    return


def ensure_dir(p: str):
    if p is None or p == "":
        return
    os.makedirs(p, exist_ok=True)


def get_slide_ids_from_features(feature_dirs: List[str]) -> List[str]:
    """Get list of slide IDs that have features in all directories.
    
    Returns only slide IDs that exist in all feature directories (for ensemble).
    """
    # Get slide IDs from first directory
    files = sorted(glob.glob(os.path.join(feature_dirs[0], "*.h5")))
    slide_ids = set([os.path.splitext(os.path.basename(f))[0] for f in files])
    
    # For ensemble, intersect with other directories to ensure all have features
    if len(feature_dirs) > 1:
        for feat_dir in feature_dirs[1:]:
            files = sorted(glob.glob(os.path.join(feat_dir, "*.h5")))
            ids = set([os.path.splitext(os.path.basename(f))[0] for f in files])
            slide_ids = slide_ids.intersection(ids)
    
    return sorted(list(slide_ids))


def main():
    parser = argparse.ArgumentParser(
        description="Generate prediction overlay visualizations for WSIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize TCGA-BRCA slides with an ensemble model
  python visualization.py \\
      --features_dir "Data/TCGA-BRCA-processed/20x_224px_70px_overlap" \\
      --wsi_dir "Data/TCGA-BRCA" \\
      --model_dir "outputs/Hibou-L,UNI2,Virchow2" \\
      --out_dir "outputs/visualizations/TCGA-BRCA" \\

  # Visualize lung slides with a single expert model
  python visualization.py \\
      --features_dir "Data/TCGA-LUAD-processed/20x_224px" \\
      --wsi_dir "Data/TCGA-LUAD" \\
      --model_dir "outputs/Virchow2" \\
      --out_dir "outputs/visualizations/TCGA-LUAD" \\
        """
    )
    
    parser.add_argument(
        "--features_dir", 
        type=str, 
        required=True,
        help="Directory containing features_* subdirectories (e.g., features_virchow2, features_uni_v2)"
    )
    parser.add_argument(
        "--wsi_dir",
        type=str,
        required=True,
        help="Directory containing original WSI files for thumbnail generation"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        required=True,
        help="Output directory for visualization images"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Override checkpoint directory (default: CKPT_DIR)"
    )
    parser.add_argument(
        "--magnification",
        type=int,
        default=None,
        help="Optional: override magnification (e.g., 20 or 40). If omitted the script will read it from the WSI properties."
    )
    parser.add_argument(
        "--slide_id",
        type=str,
        default=None,
        help="Optional: Process only this specific slide ID"
    )
    parser.add_argument(
        "--predictions_only",
        action="store_true",
        help="Skip overlay image generation; only run inference and save "
             "CSV predictions. Much faster since WSI thumbnails are not loaded."
    )
    parser.add_argument(
        "--no_legend",
        action="store_true",
        help="Do not draw the class legend on the overlay image"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.features_dir):
        print(f"Error: Features directory not found: {args.features_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.wsi_dir):
        print(f"Error: WSI directory not found: {args.wsi_dir}")
        sys.exit(1)
    
    ensure_dir(args.out_dir)
    
    ckpt_path = os.path.join(args.model_dir, "best_linear_probe.pt") \
        if args.model_dir else CKPT_PATH
    model, metadata = load_model(ckpt_path)

    is_ensemble   = metadata["is_ensemble"]
    expert_models = metadata["expert_models"]
    feature_dirs  = discover_feature_dirs(args.features_dir, expert_models)

    if is_ensemble:
        print(f"\nLoaded CrossAttentionEnsemble with {len(expert_models)} experts "
              f"(cross_attention_layer='{metadata.get('cross_attention_layer', '?')}'):")
        for i, em in enumerate(expert_models):
            print(f"  Expert {i}: {'+'.join(em)} -> {feature_dirs[i]}")
    else:
        print(f"\nLoaded single-expert model ({metadata.get('model_type', 'mlp')}): "
              f"{expert_models[0][0]} -> {feature_dirs[0]}")
    
    # Get slide IDs that have features
    if args.slide_id:
        slide_ids = [args.slide_id]
        print(f"\nProcessing single slide: {args.slide_id}")
    else:
        slide_ids = get_slide_ids_from_features(feature_dirs)
        print(f"\nFound {len(slide_ids)} slides with features")
    
    if len(slide_ids) == 0:
        print("No feature files found!")
        return
    
    # Process each slide
    pbar = tqdm(slide_ids, desc="Processing slides")
    for slide_id in pbar:
        try:
            overlay_predictions_on_wsi(
                slide_id=slide_id,
                model=model,
                metadata=metadata,
                feature_dirs=feature_dirs,
                out_dir=args.out_dir,
                wsi_dir=args.wsi_dir,
                magnification=args.magnification,
                predictions_only=args.predictions_only,
                no_legend=args.no_legend,
                pbar=pbar,
            )
        except Exception as e:
            print(f"Error visualizing {slide_id}: {e}")
    
    print(f"\nDone! Visualizations saved to: {args.out_dir}")


if __name__ == "__main__":
    main()

