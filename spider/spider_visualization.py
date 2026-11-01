"""
Generates SPIDER prediction overlay visualizations for Whole Slide Images.

Uses the same pre-computed H5 coordinate files as visualization.py for patch
locations, maps SPIDER predictions into the unified tissue taxonomy, and renders
with the same tab20 colormap / Epithelial-yellow special case.

Usage:
    python Spider/spider_visualization.py \\
        --features_dir ../Data/TCGA-SKCM-processed/20x_224px_70px_overlap \\
        --wsi_dir ../Data/TCGA-SKCM \\
        --out_dir outputs/TCGA-SKCM-SPIDER \\
        --model histai/SPIDER-skin-model

    # Single slide
    python Spider/spider_visualization.py \\
        --features_dir ../Data/TCGA-SKCM-processed/20x_224px_70px_overlap \\
        --wsi_dir ../Data/TCGA-SKCM \\
        --out_dir outputs/TCGA-SKCM-SPIDER \\
        --model histai/SPIDER-skin-model \\
        --slide_id TCGA-EB-A44O-01Z-00-DX1
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
from PIL import Image, ImageDraw
import openslide

# Add parent directory so we can import data_utils / dataset_mappings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModel, AutoProcessor

from dataset_mappings import (
    SPIDER_BREAST_TO_UNIFIED,
    SPIDER_COLORECTAL_TO_UNIFIED,
    SPIDER_SKIN_TO_UNIFIED,
    SPIDER_THORAX_TO_UNIFIED,
)
from data_utils import UNIFIED_LABELS, UNIFIED_NAME_TO_ID

# ---------------------------------------------------------------------------
# Default paths (can be overridden by CLI)
# ---------------------------------------------------------------------------
WSI_LOCATION = os.path.join("Data", "TCGA-SKCM")
OUTPUT_LOCATION = os.path.join("outputs", "TCGA-SKCM SPIDER")

DEFAULT_MODEL = "histai/SPIDER-skin-model"

# ---------------------------------------------------------------------------
# Inference / rendering config  (mirrors visualization.py values)
# ---------------------------------------------------------------------------
RADIUS_TILES = 2               # context tiles around each patch
BATCH_SIZE = 16
NUM_WORKERS = max(1, min(8, os.cpu_count() or 4))
PREFETCH = NUM_WORKERS * 4
CONFIDENCE_THRESHOLD = 0.5     # patches below → transparent / "Other" in legend
EPITHELIAL_COLOR = (255, 255, 0)

# ---------------------------------------------------------------------------
# Unified class colors  (identical to visualization.py)
# ---------------------------------------------------------------------------
NUM_UNIFIED = len(UNIFIED_LABELS)
UNIFIED_CMAP = cm.get_cmap("tab20", NUM_UNIFIED)
EPITHELIAL_IDX: int = UNIFIED_NAME_TO_ID.get("Epithelial", -1)
BACKGROUND_IDX: int = UNIFIED_NAME_TO_ID.get("Background", NUM_UNIFIED - 1)


def _unified_color(cls_idx: int) -> Tuple[int, int, int]:
    """Return RGB color for a unified class index (same logic as visualization.py)."""
    if cls_idx == EPITHELIAL_IDX:
        return EPITHELIAL_COLOR
    return tuple(
        (np.array(UNIFIED_CMAP(cls_idx)[:3]) * 255).astype(np.uint8).tolist()
    )


# ---------------------------------------------------------------------------
# SPIDER model name → unified mapping
# ---------------------------------------------------------------------------
_MODEL_TO_UNIFIED_MAP: Dict[str, dict] = {
    "skin":       SPIDER_SKIN_TO_UNIFIED,
    "thorax":     SPIDER_THORAX_TO_UNIFIED,
    "colorectal": SPIDER_COLORECTAL_TO_UNIFIED,
    "breast":     SPIDER_BREAST_TO_UNIFIED,
}


def _get_unified_mapping(model_name: str) -> dict:
    """Select the SPIDER→unified mapping based on the model identifier string."""
    lower = model_name.lower()
    for key, mapping in _MODEL_TO_UNIFIED_MAP.items():
        if key in lower:
            return mapping
    raise ValueError(
        f"Cannot infer organ from model name '{model_name}'. "
        f"Known keys: {list(_MODEL_TO_UNIFIED_MAP.keys())}. "
        "Pass --organ to override."
    )


# ---------------------------------------------------------------------------
# Feature directory → model name  (same as visualization.py)
# ---------------------------------------------------------------------------
FEATURE_DIR_TO_MODEL = {
    "features_virchow2": "Virchow2",
    "features_hoptimus0": "H-optimus-0",
    "features_hibou_l": "Hibou-L",
    "features_uni_v2": "UNI2",
    "features_conch_v1": "CONCH",
}


def _find_any_feature_dir(base_dir: str) -> str:
    """Return the first available features_* subdirectory (we only need coords)."""
    for dir_name in FEATURE_DIR_TO_MODEL:
        candidate = os.path.join(base_dir, dir_name)
        if os.path.isdir(candidate):
            return candidate
    matches = sorted(glob.glob(os.path.join(base_dir, "features_*")))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No features_* subdirectory found inside: {base_dir}"
    )


def _get_slide_ids(feature_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(feature_dir, "*.h5")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def _load_coords(feature_dir: str, slide_id: str) -> np.ndarray:
    """Load (N, 2) integer coordinates from the H5 feature file."""
    h5_path = os.path.join(feature_dir, f"{slide_id}.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        return f["coords"][:].astype(int)  # (N, 2) in level-0 pixel space


# ---------------------------------------------------------------------------
# OpenSlide helpers  (same as visualization.py)
# ---------------------------------------------------------------------------
def _fetch_mpp(img: openslide.OpenSlide) -> float:
    mpp_keys = [
        openslide.PROPERTY_NAME_MPP_X,
        "openslide.mirax.MPP",
        "aperio.MPP",
        "hamamatsu.XResolution",
    ]
    for key in mpp_keys:
        if key in img.properties:
            try:
                return round(float(img.properties[key]), 4)
            except ValueError:
                continue
    x_res = img.properties.get("tiff.XResolution")
    unit  = img.properties.get("tiff.ResolutionUnit")
    if x_res and unit:
        try:
            if unit.lower() == "centimeter":
                return round(10000 / float(x_res), 4)
            elif unit.upper() == "INCH":
                return round(25400 / float(x_res), 4)
        except ValueError:
            pass
    raise ValueError("Unable to extract MPP from slide metadata")


def _fetch_magnification(img: openslide.OpenSlide) -> int:
    mpp = _fetch_mpp(img)
    if mpp < 0.16:  return 80
    if mpp < 0.2:   return 60
    if mpp < 0.3:   return 40
    if mpp < 0.6:   return 20
    if mpp < 1.2:   return 10
    raise ValueError(f"Unexpectedly low MPP={mpp}")


def _find_wsi(slide_id: str, wsi_dir: str) -> Optional[str]:
    for root, _dirs, files in os.walk(wsi_dir):
        for fname in files:
            if slide_id in fname:
                return os.path.join(root, fname)
    return None


def _read_thumbnail(wsi_path: str, max_side: int = 3000) -> Tuple[Image.Image, Tuple[int, int]]:
    sl = openslide.OpenSlide(wsi_path)
    sw, sh = sl.dimensions
    scale = min(1.0, max_side / max(sw, sh))
    tw, th = max(1, int(sw * scale)), max(1, int(sh * scale))
    try:
        thumb = sl.get_thumbnail((tw, th)).convert("RGB")
    except Exception:
        thumb = sl.read_region((0, 0), 0, (sw, sh)).convert("RGB").resize(
            (tw, th), resample=Image.BILINEAR
        )
    finally:
        sl.close()
    return thumb, (sw, sh)


def _read_region_padded(
    sl: openslide.OpenSlide, x: int, y: int, w: int, h: int,
    pad_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    sw, sh = sl.dimensions
    x0, y0, x1, y1 = int(x), int(y), int(x) + int(w), int(y) + int(h)
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(sw, x1), min(sh, y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return Image.new("RGB", (int(w), int(h)), pad_color)
    crop = sl.read_region((ix0, iy0), 0, (ix1 - ix0, iy1 - iy0)).convert("RGB")
    canvas = Image.new("RGB", (int(w), int(h)), pad_color)
    canvas.paste(crop, (ix0 - x0, iy0 - y0))
    return canvas


def _read_context_patch(
    sl: openslide.OpenSlide, x: int, y: int, tile_w: int, tile_h: int,
    radius: int = RADIUS_TILES,
) -> Image.Image:
    """Read a (2*radius+1) × (2*radius+1) neighbourhood of tiles as one big image."""
    big_w = tile_w * (2 * radius + 1)
    big_h = tile_h * (2 * radius + 1)
    start_x = x - radius * tile_w
    start_y = y - radius * tile_h
    return _read_region_padded(sl, start_x, start_y, big_w, big_h)


# ---------------------------------------------------------------------------
# SPIDER predictor
# ---------------------------------------------------------------------------
class SpiderPredictor:
    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SPIDER model '{model_name}' on {self.device}")
        base = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = base.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        cfg = getattr(self.model, "config", None)
        self.class_names: List[str] = getattr(cfg, "class_names", None) or []
        self.num_classes: int = (
            len(self.class_names) if self.class_names
            else int(getattr(cfg, "num_labels", 24))
        )

    @torch.inference_mode()
    def predict_batch(
        self, images: List[Image.Image]
    ) -> Tuple[List[str], np.ndarray]:
        """Return (predicted_class_names, logits array [N, C])."""
        if not images:
            return [], np.zeros((0, self.num_classes), dtype=np.float32)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        return out.predicted_class_names, out.logits.cpu().numpy()


# ---------------------------------------------------------------------------
# Per-slide visualization
# ---------------------------------------------------------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def visualize_slide(
    slide_id: str,
    predictor: SpiderPredictor,
    unified_mapping: dict,
    feature_dir: str,
    wsi_dir: str,
    out_dir: str,
    magnification: Optional[int] = None,
) -> None:
    """Predict and render one slide; save overlay PNG + predictions CSV."""
    out_path = os.path.join(out_dir, f"{slide_id}_spider_overlay.png")
    if os.path.exists(out_path):
        print(f"Skipping {slide_id} (already exists)")
        return

    print(f"Visualizing {slide_id}")

    # -- coords from H5 --
    coords = _load_coords(feature_dir, slide_id)  # (N, 2)
    xs, ys = coords[:, 0], coords[:, 1]

    # -- WSI --
    wsi_path = _find_wsi(slide_id, wsi_dir)
    if wsi_path is None:
        print(f"  WARNING: WSI not found for {slide_id}, skipping")
        return

    sl = openslide.OpenSlide(wsi_path)
    sw, sh = sl.dimensions

    mag = magnification if magnification is not None else _fetch_magnification(sl)
    tile_px = int(round((mag / 20) * 224))   # tile footprint in level-0 pixels
    print(f"  Magnification: {mag}x  →  tile_px={tile_px}")

    thumb, _ = _read_thumbnail(wsi_path)
    tw, th = thumb.size

    overlay = thumb.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    # -- batched inference --
    all_x:       List[int]   = []
    all_y:       List[int]   = []
    all_spider:  List[str]   = []   # raw SPIDER class name
    all_unified: List[str]   = []
    all_conf:    List[float] = []
    all_logits:  List[np.ndarray] = []

    batch_imgs: List[Image.Image] = []
    batch_meta: List[Tuple[int, int]] = []   # (x, y) at level-0

    def _flush(imgs: List[Image.Image], meta: List[Tuple[int, int]]) -> None:
        spider_names, logits = predictor.predict_batch(imgs)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        confs = probs.max(axis=1)

        for (bx, by), spider_cls, logit_vec, conf in zip(meta, spider_names, logits, confs):
            unified_cls = unified_mapping.get(spider_cls, "Background")
            unified_idx = UNIFIED_NAME_TO_ID.get(unified_cls, BACKGROUND_IDX)

            all_x.append(bx)
            all_y.append(by)
            all_spider.append(spider_cls)
            all_unified.append(unified_cls)
            all_conf.append(float(conf))
            all_logits.append(logit_vec)

            # Draw on overlay
            tx = int(bx / sw * tw)
            ty = int(by / sh * th)
            twr = max(1, int(tile_px / sw * tw))
            thr = max(1, int(tile_px / sh * th))

            if conf < CONFIDENCE_THRESHOLD:
                alpha = 0
                color = (0, 0, 0)
            else:
                color = _unified_color(unified_idx)
                alpha = int(conf * 255)
            draw.rectangle(
                [tx, ty, tx + twr, ty + thr],
                fill=(color[0], color[1], color[2], alpha),
            )

    def _read_fn(coord: Tuple[int, int]) -> Tuple[Image.Image, Tuple[int, int]]:
        bx, by = coord
        img = _read_context_patch(sl, bx, by, tile_px, tile_px, RADIUS_TILES)
        return img, (bx, by)

    n = len(xs)
    coords_list = list(zip(xs.tolist(), ys.tolist()))

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for chunk_start in range(0, n, PREFETCH):
            chunk = coords_list[chunk_start : chunk_start + PREFETCH]
            futs = [ex.submit(_read_fn, c) for c in chunk]
            for i, fut in enumerate(as_completed(futs)):
                img, meta = fut.result()
                print(f"  Tile {chunk_start + i + 1}/{n}", end="\r")
                batch_imgs.append(img)
                batch_meta.append(meta)
                if len(batch_imgs) >= BATCH_SIZE:
                    _flush(batch_imgs, batch_meta)
                    batch_imgs.clear()
                    batch_meta.clear()

    if batch_imgs:
        _flush(batch_imgs, batch_meta)

    sl.close()
    print()  # newline after \r progress

    # -- legend (unified classes, same ordering / colors as visualization.py) --
    legend_items = NUM_UNIFIED
    legend_h, legend_w = 18, 260
    # +1 row for "Other" (low-confidence)
    legend = Image.new("RGBA", (legend_w, (legend_items + 1) * legend_h), (255, 255, 255, 220))
    ld = ImageDraw.Draw(legend)
    for i in range(legend_items):
        color = _unified_color(i)
        y0 = i * legend_h
        ld.rectangle([4, y0 + 4, 18, y0 + 14], fill=color)
        ld.text((24, y0 + 2), UNIFIED_LABELS[i], fill=(0, 0, 0, 255))
    y0 = legend_items * legend_h
    ld.rectangle([4, y0 + 4, 18, y0 + 14], fill=(128, 128, 128))
    ld.text((24, y0 + 2), "Other", fill=(0, 0, 0, 255))
    overlay.paste(legend, (6, 6), legend)

    # -- save overlay --
    _ensure_dir(out_dir)
    Image.alpha_composite(thumb.convert("RGBA"), overlay).save(out_path)
    print(f"  Saved overlay → {out_path}")

    # -- save CSV --
    logits_arr = np.array(all_logits)  # (N, C_spider)
    data: dict = {
        "x": all_x,
        "y": all_y,
        "spider_prediction": all_spider,
        "unified_prediction": all_unified,
        "confidence": all_conf,
    }
    for i, cls_name in enumerate(predictor.class_names):
        data[f"logit_{cls_name}"] = logits_arr[:, i].tolist()

    csv_dir = os.path.join(out_dir, "CSV")
    _ensure_dir(csv_dir)
    csv_path = os.path.join(csv_dir, f"{slide_id}_spider_predictions.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)
    print(f"  Saved predictions → {csv_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPIDER overlay visualization using H5 patch coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--features_dir", type=str, required=True,
        help="Directory with features_* subdirs (only coords are read, not embeddings)",
    )
    parser.add_argument(
        "--wsi_dir", type=str, required=True,
        help="Directory containing the original WSI files (.svs/.tiff/…)",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for overlay PNGs and prediction CSVs",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model ID for SPIDER (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--organ", type=str, default=None,
        choices=list(_MODEL_TO_UNIFIED_MAP.keys()),
        help="Organ type for SPIDER→unified mapping. Inferred from --model if omitted.",
    )
    parser.add_argument(
        "--magnification", type=int, default=None,
        help="Override slide magnification (e.g. 20 or 40). Auto-detected from WSI if omitted.",
    )
    parser.add_argument(
        "--slide_id", type=str, default=None,
        help="Process only this specific slide ID (default: all slides in features_dir)",
    )
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")

    if not os.path.isdir(args.features_dir):
        print(f"Error: features_dir not found: {args.features_dir}")
        sys.exit(1)
    if not os.path.isdir(args.wsi_dir):
        print(f"Error: wsi_dir not found: {args.wsi_dir}")
        sys.exit(1)

    # Resolve which feature subdirectory to use for coordinates
    feature_dir = _find_any_feature_dir(args.features_dir)
    print(f"Using coordinates from: {feature_dir}")

    # Resolve SPIDER → unified mapping
    if args.organ is not None:
        unified_mapping = _MODEL_TO_UNIFIED_MAP[args.organ]
        print(f"Using explicit organ mapping: {args.organ}")
    else:
        unified_mapping = _get_unified_mapping(args.model)
        print(f"Inferred organ mapping from model name '{args.model}'")

    # Build predictor
    predictor = SpiderPredictor(args.model)
    print(
        f"SPIDER classes ({predictor.num_classes}): "
        f"{', '.join(predictor.class_names[:8])}{'…' if predictor.num_classes > 8 else ''}"
    )

    # Slide list
    if args.slide_id:
        slide_ids = [args.slide_id]
    else:
        slide_ids = _get_slide_ids(feature_dir)
    print(f"Slides to process: {len(slide_ids)}")

    if not slide_ids:
        print("No slides found — check --features_dir.")
        return

    _ensure_dir(args.out_dir)

    for slide_id in slide_ids:
        try:
            visualize_slide(
                slide_id=slide_id,
                predictor=predictor,
                unified_mapping=unified_mapping,
                feature_dir=feature_dir,
                wsi_dir=args.wsi_dir,
                out_dir=args.out_dir,
                magnification=args.magnification,
            )
        except Exception as exc:
            print(f"  ERROR processing {slide_id}: {exc}")

    print(f"\nDone. Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
