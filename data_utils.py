from typing import Dict, List, Tuple

import torch
import os
import random
import numpy as np

from PIL import Image, ImageDraw
from tqdm import tqdm

from torch.utils.data import Dataset

from spider.spider_dataset import SPIDERDataset

from dataset_mappings import *

import csv

UNIFIED_LABELS = [
    "Tumor",
    "Fat",
    "Stroma",
    "Inflammatory", 
    "Necrosis",
    "Nerve",
    "Mucin",
    "Vascular",
    "Skin adnexa",
    "Epithelial",
    "Background"
]
UNIFIED_NAME_TO_ID = {name: idx for idx, name in enumerate(UNIFIED_LABELS)}
UNIFIED_ID_TO_NAME = {idx: name for idx, name in enumerate(UNIFIED_LABELS)}

ORGANS = [
    "breast",
    "skin",
    "thorax",
    "colorectal",
    "lung",
    "prostate"
]

DATA_ROOT = "Data"
BCSS_DATA_ROOT = os.path.join(DATA_ROOT, "BCSS")
SPIDER_SKIN_DATA_ROOT = os.path.join(DATA_ROOT, "SPIDER-skin", "SPIDER-skin")
SPIDER_THORAX_DATA_ROOT = os.path.join(DATA_ROOT, "SPIDER-thorax", "SPIDER-thorax")
SPIDER_BREAST_DATA_ROOT = os.path.join(DATA_ROOT, "SPIDER-breast", "SPIDER-breast")
SPIDER_COLORECTAL_DATA_ROOT = os.path.join(DATA_ROOT, "SPIDER-colorectal", "SPIDER-colorectal")
LUB_LUNG_DATA_ROOT = os.path.join(DATA_ROOT, "LubLung", "LubLung")
KATHER_DATA_ROOT = os.path.join(DATA_ROOT, "Kather", "NCT-CRC-HE-100K-NONORM")
KATHER_5000_DATA_ROOT = os.path.join(DATA_ROOT, "Kather-5000")
PROSTATE_ROOT = os.path.join(DATA_ROOT, "Prostate")

UNIFIED_DATA_ROOT = os.path.join(DATA_ROOT, "Unified Dataset")

IMAGES_DIR = os.path.join(BCSS_DATA_ROOT, "images")
MASKS_DIR = os.path.join(BCSS_DATA_ROOT, "masks")
META_TSV = os.path.join(BCSS_DATA_ROOT, "meta", "gtruth_codes.tsv")

TILE_ROOT = os.path.join(UNIFIED_DATA_ROOT, "tiles224")
INDEX_CSV = os.path.join(TILE_ROOT, "index.csv")

SEED_IDX = 0
SEEDS = [1337, 7555, 2, 325, 32843826]
SEED = SEEDS[SEED_IDX]
TILE_SIZE = 224
MAJORITY_THRESHOLD = 0.5 


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_label_map(tsv_path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2name: Dict[int, str] = {}
    with open(tsv_path, "r") as f:
        _ = f.readline() # skip header
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")

            if len(parts) != 2:
                raise ValueError(f"Malformed TSV line: {line}")
            
            label_name, code_str = parts
            code = int(code_str)
            id2name[code] = label_name

    return id2name


def tile_image_and_mask(img: Image.Image, mask: Image.Image, tile_size: int) -> List[Tuple[int, int, Image.Image, int]]:
    """Return list of (x, y, img_crop, mask_crop) for non-overlapping tiles fully inside the image.
       Discards right/bottom edges that don't fit tile_size exactly.
    """
    W, H = img.size
    w_tiles = W // tile_size
    h_tiles = H // tile_size
    tiles = []
    for ty in range(h_tiles):
        for tx in range(w_tiles):
            x0 = tx * tile_size
            y0 = ty * tile_size
            crop_box = (x0, y0, x0 + tile_size, y0 + tile_size)
            mask_crop = mask.crop(crop_box)
            keep, label_id = determine_majority_label(mask_crop, MAJORITY_THRESHOLD)

            if keep and label_id != 0:
                tiles.append((tx, ty, img.crop(crop_box), label_id))
    return tiles


def determine_majority_label(mask_crop: Image.Image, majority_threshold: float) -> Tuple[bool, int]:
    """Return (keep, label_id). keep=True iff a class covers > majority_threshold of pixels.
       mask is assumed to be single-channel with pixel values as GT_code integers.
    """
    arr = np.array(mask_crop, dtype=np.int64)
    total = arr.size
    # Fast bincount over non-negative integers
    counts = np.bincount(arr.ravel())
    label_id = int(np.argmax(counts))
    frac = counts[label_id] / float(total)
    keep = frac > majority_threshold
    return keep, label_id


def build_bcss_tiles(rows: List[List[str]]):
    id2name = read_label_map(META_TSV)
    ensure_dir(TILE_ROOT)

    image_files = sorted(os.listdir(IMAGES_DIR))
    print(f"Found {len(image_files)} source images in {IMAGES_DIR}")

    kept = 0

    for fname in image_files:
        img_path = os.path.join(IMAGES_DIR, fname)
        mask_path = os.path.join(MASKS_DIR, fname)
        if not os.path.exists(mask_path):
            print(f"[WARN] No mask for {fname}, skipping.")
            continue

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if mask.mode != "L":
            mask = mask.convert("L")
        if img.size != mask.size:
            print(f"[WARN] Size mismatch for {fname}: image {img.size}, mask {mask.size}. Skipping.")
            continue

        tiles = tile_image_and_mask(img, mask, TILE_SIZE)
        base, _ = os.path.splitext(fname)

        for tx, ty, img_crop, label_id in tiles:
            orig_label_name = id2name[label_id]
            if orig_label_name not in BCSS_TO_UNIFIED: # Cannot make unified label, skip
                print(f"Skipping BCSS label {orig_label_name} not in mapping")
                continue
            
            unified_name = BCSS_TO_UNIFIED[orig_label_name]
            unified_id = UNIFIED_NAME_TO_ID[unified_name]
            class_dir = os.path.join(TILE_ROOT, unified_name)
            ensure_dir(class_dir)
            out_name = f"{base}_x{tx}_y{ty}.png"
            out_path = os.path.join(class_dir, out_name)
            img_crop.save(out_path)

            rows.append([out_path, str(unified_id), unified_name, "BCSS", orig_label_name, "breast"])
            kept += 1

    print(f"BCSS tiling complete. Kept {kept} tiles.")

    return rows, kept


def build_spider_tiles(rows: List[List[str]], path_root: str, dataset_to_unified: Dict[str, str], dataset_name: str):
    dataset = SPIDERDataset(path_root, context_size=1, transform=None)
    ensure_dir(TILE_ROOT)

    print(f"Found {len(dataset)} source images in SPIDER dataset")

    kept = 0

    for i in range(len(dataset)):
        img, label_id = dataset[i]
        orig_label_name = dataset.label_to_class[label_id]

        if orig_label_name not in dataset_to_unified: # Cannot make unified label, skip
            print(f"Skipping SPIDER label {orig_label_name} not in mapping")
            continue
        unified_name = dataset_to_unified[orig_label_name]
        unified_id = UNIFIED_NAME_TO_ID[unified_name]
        class_dir = os.path.join(TILE_ROOT, unified_name)
        ensure_dir(class_dir)
        out_name = f"spider_{dataset_name}_{i:05d}.png"
        out_path = os.path.join(class_dir, out_name)
        img.save(out_path)

        rows.append([out_path, str(unified_id), unified_name, f"SPIDER_{dataset_name}", orig_label_name, dataset_name])
        kept += 1

    print(f"SPIDER_{dataset_name} tiling complete. Kept {kept} tiles.")
    
    return rows, kept


def build_lublung_tiles(rows: List[List[str]], path_root: str, dataset_to_unified: Dict[str, str]):
    ensure_dir(TILE_ROOT)

    kept = 0

    for class_folder in os.listdir(path_root):
        class_path = os.path.join(path_root, class_folder)
        if not os.path.isdir(class_path):
            continue

        for fname in os.listdir(class_path):
            file_path = os.path.join(class_path, fname)
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Invalid file type found: {fname}")
                continue

            if class_folder not in dataset_to_unified:
                print(f"Skipping LubLung label {class_folder} not in mapping")
                continue

            unified_name = dataset_to_unified[class_folder]
            unified_id = UNIFIED_NAME_TO_ID[unified_name]
            class_dir = os.path.join(TILE_ROOT, unified_name)
            ensure_dir(class_dir)

            out_name = f"lublung_{class_folder}_{fname}"
            out_path = os.path.join(class_dir, out_name)
            
            img = Image.open(file_path).convert("RGB")
            img.save(out_path)

            rows.append([out_path, str(unified_id), unified_name, "LubLung", class_folder, "lung"])
            kept += 1

    print(f"LubLung tiling complete. Kept {kept} tiles.")
    return rows, kept


def build_kather_tiles(rows: List[List[str]], path_root: str, dataset_to_unified: Dict[str, str]):
    """Process Kather dataset patches (.tif files) and convert to PNG.
    
    The Kather dataset has one subfolder per class with 224x224 .tif patches.
    """
    ensure_dir(TILE_ROOT)

    kept = 0

    for class_folder in os.listdir(path_root):
        class_path = os.path.join(path_root, class_folder)
        if not os.path.isdir(class_path):
            continue

        if class_folder not in dataset_to_unified:
            print(f"Skipping Kather label {class_folder} not in mapping")
            continue

        unified_name = dataset_to_unified[class_folder]
        unified_id = UNIFIED_NAME_TO_ID[unified_name]
        class_dir = os.path.join(TILE_ROOT, unified_name)
        ensure_dir(class_dir)

        # Process all .tif files in this class folder
        tif_files = [f for f in os.listdir(class_path) if f.lower().endswith('.tif')]
        print(f"Processing {len(tif_files)} patches from Kather class {class_folder}")

        for fname in tif_files:
            file_path = os.path.join(class_path, fname)
            
            # Convert .tif to .png
            base_name = os.path.splitext(fname)[0]
            out_name = f"kather_{class_folder}_{base_name}.png"
            out_path = os.path.join(class_dir, out_name)
            
            # Load and save as PNG
            img = Image.open(file_path).convert("RGB")
            img.save(out_path)

            rows.append([out_path, str(unified_id), unified_name, "Kather", class_folder, "colorectal"])
            kept += 1

    print(f"Kather tiling complete. Kept {kept} tiles.")
    return rows, kept


def build_prostate_tiles(rows: List[List[str]], path_root: str, dataset_to_unified: Dict[str, str]):
    ensure_dir(TILE_ROOT)

    kept = 0

    # First pass: count total files to process
    total_files = 0
    for class_folder in os.listdir(path_root):
        class_path = os.path.join(path_root, class_folder)
        if not os.path.isdir(class_path):
            continue
        if class_folder not in dataset_to_unified:
            continue
        total_files += len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])

    # Process files with progress bar
    with tqdm(total=total_files, desc="Processing prostate tiles", unit="file") as pbar:
        for class_folder in os.listdir(path_root):
            class_path = os.path.join(path_root, class_folder)
            if not os.path.isdir(class_path):
                continue

            if class_folder not in dataset_to_unified:
                print(f"Skipping Prostate label {class_folder} not in mapping")
                continue

            unified_name = dataset_to_unified[class_folder]
            unified_id = UNIFIED_NAME_TO_ID[unified_name]
            class_dir = os.path.join(TILE_ROOT, unified_name)
            ensure_dir(class_dir)

            # Process common image file types inside this class folder
            for fname in os.listdir(class_path):
                file_path = os.path.join(class_path, fname)
                
                img = Image.open(file_path).convert("RGB")

                base_name = os.path.splitext(fname)[0]
                out_name = f"prostate_{class_folder}_{base_name}.png"
                out_path = os.path.join(class_dir, out_name)
                
                img.save(out_path)

                # Use 'prostate' as the organ identifier for these rows
                rows.append([out_path, str(unified_id), unified_name, "Prostate", class_folder, "prostate"])
                kept += 1
                pbar.update(1)

    print(f"Prostate tiling complete. Kept {kept} tiles.")
    return rows, kept


def generate_synthetic_background():
    # Generate 40 synthetic background patches with sparse pink stains
    ensure_dir(TILE_ROOT)
    synth_dir = os.path.join(TILE_ROOT, "Background")
    ensure_dir(synth_dir)

    num_patches = 250
    width = TILE_SIZE
    height = TILE_SIZE
    background_color = (255, 255, 255)
    pink_colors = [
        (255, 192, 203),  # light pink
        (250, 170, 185),
        (245, 160, 175),
    ]

    stain_min = 0   # minimum number of small stains
    stain_max = 5  # maximum number of small stains
    max_radius = 5  # max radius for circular stains

    new_rows: List[List[str]] = []
    unified_id = UNIFIED_NAME_TO_ID["Background"]
    unified_name = "Background"

    for i in range(num_patches):
        img = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(img)
        # Place random pink oval stains (random width/height)
        n_stains = random.randint(stain_min, stain_max)
        for _ in range(n_stains):
            cx = random.randint(0, width - 1)
            cy = random.randint(0, height - 1)
            rx = random.randint(2, max_radius)
            ry = random.randint(2, max_radius)
            color = random.choice(pink_colors)

            x0 = max(0, cx - rx)
            y0 = max(0, cy - ry)
            x1 = min(width, cx + rx)
            y1 = min(height, cy + ry)
            if x1 > x0 and y1 > y0:
                draw.ellipse([x0, y0, x1, y1], fill=color)

        out_img = img
        fname = f"synthetic_background_{i:03d}.png"
        fpath = os.path.join(synth_dir, fname)
        out_img.save(fpath)
        # Append row: path, unified_label_id, unified_label_name, source, original_label
        new_rows.append([fpath, str(unified_id), unified_name, "Synthetic", "breast"])

    # Generate 250 pure white background patches
    num_white_patches = 250
    for i in range(num_white_patches):
        img = Image.new("RGB", (width, height), background_color)  # Pure white background
        fname = f"synthetic_white_background_{i:03d}.png"
        fpath = os.path.join(synth_dir, fname)
        img.save(fpath)
        # Append row: path, unified_label_id, unified_label_name, source, original_label
        new_rows.append([fpath, str(unified_id), unified_name, "Synthetic", "breast"])

    with open(INDEX_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    print(f"Added {len(new_rows)} synthetic background patches to {synth_dir} and updated index {INDEX_CSV}.")



def build_tiles_and_index():
    rows: List[List[str]] = []
    kept = 0

    rows, kept_bcss = build_bcss_tiles(rows)
    kept += kept_bcss
    
    rows, kept_spider_skin = build_spider_tiles(rows, SPIDER_SKIN_DATA_ROOT, SPIDER_SKIN_TO_UNIFIED, "skin")
    kept += kept_spider_skin

    rows, kept_spider_thorax = build_spider_tiles(rows, SPIDER_THORAX_DATA_ROOT, SPIDER_THORAX_TO_UNIFIED, "thorax")
    kept += kept_spider_thorax

    rows, kept_spider_colorectal = build_spider_tiles(rows, SPIDER_COLORECTAL_DATA_ROOT, SPIDER_COLORECTAL_TO_UNIFIED, "colorectal")
    kept += kept_spider_colorectal

    rows, kept_spider_breast = build_spider_tiles(rows, SPIDER_BREAST_DATA_ROOT, SPIDER_BREAST_TO_UNIFIED, "breast")
    kept += kept_spider_breast

    rows, kept_lublung = build_lublung_tiles(rows, LUB_LUNG_DATA_ROOT, LUB_LUNG_TO_UNIFIED)
    kept += kept_lublung

    rows, kept_kather = build_kather_tiles(rows, KATHER_DATA_ROOT, KATHER_TO_UNIFIED)
    kept += kept_kather

    rows, kept_kather_5000 = build_kather_tiles(rows, KATHER_5000_DATA_ROOT, KATHER_5000_TO_UNIFIED)
    kept += kept_kather_5000

    # Write index
    with open(INDEX_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "unified_label_id", "unified_label_name", "source", "original_label", "organ"])
        writer.writerows(rows)

    print(f"Tiling complete. Kept {kept} tiles. Index at {INDEX_CSV}")

    generate_synthetic_background()


# ---------------------------
# Dataset
# ---------------------------


class TileDataset(Dataset):
    def __init__(self, index_rows: List[List[str]], transform):
        self.rows = index_rows
        self.transform = transform


    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = int(label)
        return img, label
    

class EmbeddingDataset(torch.utils.data.Dataset):
    """Wrap precomputed embeddings for use with DataLoader.

    Expects a dict with 'X' (Tensor [N,D]) and 'y' (Tensor [N]).
    """
    def __init__(self, data: Dict):
        self.X = data["X"]
        self.y = data["y"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx])


class MultiEmbeddingDataset(torch.utils.data.Dataset):
    """Wrap multiple precomputed embeddings for ensemble training.
    
    Expects a dict with:
      'X': List of Tensors, one per expert, each of shape [N, D_i]
      'y': Tensor [N] with labels
      'organ': (Optional) Tensor [N] with organ indices
    
    Returns a tuple of (list of embeddings, label, [organ_idx]) where the list contains
    one embedding tensor per expert for the given sample.
    """
    def __init__(self, data: Dict):
        self.X_list = data["X"]  # List of tensors
        self.y = data["y"]
        self.organ = data.get("organ", None)
        
        # Validate all experts have same number of samples
        n_samples = self.X_list[0].shape[0]
        for i, X in enumerate(self.X_list):
            if X.shape[0] != n_samples:
                raise ValueError(f"Expert {i} has {X.shape[0]} samples, expected {n_samples}")
        
    def __len__(self):
        return self.X_list[0].shape[0]

    def __getitem__(self, idx):
        out_x = [X[idx] for X in self.X_list]
        out_y = int(self.y[idx])
        if self.organ is not None:
            return out_x, out_y, int(self.organ[idx])
        return out_x, out_y


if __name__ == "__main__":
    rows, kept = build_prostate_tiles([], PROSTATE_ROOT, PROSTATE_TO_UNIFIED)

    with open(INDEX_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "unified_label_id", "unified_label_name", "source", "original_label", "organ"])
        writer.writerows(rows)
