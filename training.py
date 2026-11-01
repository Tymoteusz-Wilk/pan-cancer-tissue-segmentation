"""
File: Model Training/training.py

What this script does:
1) Reads BCSS images + pixel masks
2) Tiles each large patch into 224x224 crops. A crop is KEPT iff >50% of its pixels share the same class.
   - The crop gets the label of that majority class; otherwise it's discarded.
   - Crops are saved to Data/BCSS/tiles224/<class_name>/<image_id>_x{xi}_y{yi}.png
   - An index CSV is written to Data/BCSS/tiles224/index.csv with columns [path,label,label_name,source_image,x,y].
3) Builds a PyTorch Dataset that loads crops and applies the exact Virchow2 preprocessing.
4) Loads Virchow2 as a frozen backbone (via timm from Hugging Face), extracts a 2560-dim embedding (class token + mean of patch tokens) per crop,
   and trains a linear head for per-patch classification ("linear probing").
5) Trains/validates, saves the best classifier to Model Training/outputs/best_linear_probe.pt, and prints metrics.

References for Virchow2 usage (model card): https://huggingface.co/paige-ai/Virchow2 
"""

import os
import argparse
import csv
import json
import time
import random
from collections import Counter
from typing import Dict, List, Tuple

from data_utils import *
from embeddings_utils import precompute_embeddings
from embeddings_utils import EMBEDDINGS_DIR
from models import *

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt

import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from timm.layers import SwiGLUPacked

from conch.open_clip_custom import create_model_from_pretrained

from transformers import AutoImageProcessor, AutoModel


# Train setup
VAL_FRACTION = 0.0
VAL_ORGANS = ["lung"]  # Organs to hold out as a TEST set (excluded from training), e.g. ["lung"] or ["lung", "breast"].
               # VAL_FRACTION still governs the train/val split on the remaining data.
               # Set VAL_FRACTION=0.0 to train on all non-test data with no validation.
TEST_FRACTION = 0.0  # Stratified fraction of the dataset to hold out as a test set (e.g. 0.5 for 50%).
               # Applied after VAL_ORGANS exclusion. Set to 0.0 to disable.
TEST_ORGAN = "lung"  # Single organ to evaluate on after training (must be in VAL_ORGANS, or "" to evaluate all held-out organs together).

TRAIN_SUBSET_RATIO = 1.0  # Fraction of training data to use (e.g. 0.2 for 20%). Set to 1.0 for full training.
MIN_CLASS_SAMPLES = 10
BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 4)

# Stage 1: Train individual expert heads
STAGE1_EPOCHS = 10
STAGE1_LEARNING_RATE = 3e-3
STAGE1_WEIGHT_DECAY = 0.0  # try standard AdamW decay (0.01)

# Stage 2: Fine-tune ensemble with HiddenCrossAttention
STAGE2_EPOCHS = 5
STAGE2_LEARNING_RATE = 5e-4   # CA params (new, need to learn)
STAGE2_LR_EXISTING  = 5e-5   # existing encoder/classifier params (fine-tune gently)

USE_ALTERNATING = True    # False = train all experts jointly each batch (ablation)
USE_LOGIT_MLP_ABLATION = False  # True = use MLP on logits instead of Cross-Attention (ablation)
USE_GATED_AVERAGE_ABLATION = False  # True = use sample-dependent gated averaging over expert logits
USE_CONFIDENCE_WEIGHTED_ABLATION = False  # True = weight experts by their own predictive confidence
USE_ORGAN_EMBEDDING_ABLATION = False # True = use organ embedding concatenated to input (ablation)
USE_ONE_STAGE_ABLATION = False

ONE_STAGE_EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS

# If set to a path (e.g. "Model Training/outputs/MyModel/stage1_linear_probe.pt"),
# Stage 1 training is skipped and the checkpoint is loaded directly.
STAGE1_CHECKPOINT: str = None

# Ensemble Architecture Configuration
# FMoE Configuration
FMOE_NUM_HEADS = 4           # Attention heads for cross-expert attention
FMOE_NUM_ATTN_LAYERS = 2     # Number of cross-expert attention layers
FMOE_ROUTER_DIM = 64         # Router projection dimension
DIVERSITY_WEIGHT = 0.1       # Weight for expert diversity/load-balancing loss

# GatedEnsemble Configuration (Residual Connections)
# No additional hyperparameters needed - alpha is learned automatically

OUTPUT_DIR = os.path.join("PanCancerSegmentation", "outputs")

# Ensemble Configuration:
# - List of lists: Train ensemble with learned gating
#   Each sublist is a separate expert model
MODEL_NAMES = [
    ["Virchow2"],
    ["H-optimus-0"]
]

# Single-expert model type (only applies when len(MODEL_NAMES) == 1):
#   "mlp"    — encoder (Linear→GELU→Linear) as before
#   "logreg" — single Linear layer, i.e. multinomial logistic regression baseline
MODEL_TYPE = "mlp"

# ---------------------------
# Training helpers
# ---------------------------

def filter_rows_by_sources(index_rows: List[List[str]], excluded_sources: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
    """Split rows into kept and excluded partitions based on the source column."""
    if not excluded_sources:
        return index_rows, []

    excluded = {src.lower().strip() for src in excluded_sources}
    kept_rows: List[List[str]] = []
    removed_rows: List[List[str]] = []
    for row in index_rows:
        source = row[3].lower().strip()
        if source in excluded:
            removed_rows.append(row)
        else:
            kept_rows.append(row)
    return kept_rows, removed_rows


def get_ensemble_method_name() -> str:
    if USE_LOGIT_MLP_ABLATION:
        return "logit_mlp"
    if USE_GATED_AVERAGE_ABLATION:
        return "gated_average"
    if USE_CONFIDENCE_WEIGHTED_ABLATION:
        return "confidence_weighted"
    if USE_ORGAN_EMBEDDING_ABLATION:
        return "organ_cross_attention"
    return "cross_attention"

def split_index(index_rows: List[List[str]], val_fraction: float, test_organs: List[str] = None, test_fraction: float = 0.0) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """
    Split index rows into (train_rows, val_rows, test_rows).

    Two independent mechanisms for creating a test set (may be combined):
      - test_organs: all samples from the given organs are placed in the test set.
      - test_fraction: a stratified fraction of the remaining samples is placed in the test set.

    Order: organ exclusion runs first; test_fraction is then applied to whatever remains.
    Set val_fraction=0.0 to put all non-test samples in training (no validation set).

    CSV format assumed: path, unified_label_id, unified_label_name, source, original_label, organ

    Returns: (train_rows, val_rows, test_rows)  — test_rows is [] when neither mechanism is active.
    """
    test_rows: List[List[str]] = []

    if test_organs:
        test_organs_lower = {o.lower().strip() for o in test_organs}
        non_test_rows: List[List[str]] = []
        for r in index_rows:
            row_organ = r[5].lower().strip()
            if row_organ in test_organs_lower:
                test_rows.append(r[:2])
            else:
                non_test_rows.append(r)

        found_organs = {r[5].lower().strip() for r in index_rows if r[5].lower().strip() in test_organs_lower}
        for organ in test_organs:
            if organ.lower().strip() not in found_organs:
                print(f"[WARN] No samples found for test organ '{organ}'!")
        print(f"Index splitting: {test_organs} → test set ({len(test_rows)} samples).")

        # Recursively apply the stratified logic (+ optional fraction test split) to the non-test rows.
        train_rows, val_rows, fraction_test_rows = split_index(non_test_rows, val_fraction, test_organs=None, test_fraction=test_fraction)
        test_rows.extend(fraction_test_rows)
        random.shuffle(test_rows)
        return train_rows, val_rows, test_rows

    # stratified split by label (Original Logic)
    by_label: Dict[int, List[List[str]]] = {}
    for r in index_rows:
        r = r[:2]  # only need path + label
        lbl = int(r[1])
        by_label.setdefault(lbl, []).append(r)
    
    for lbl in sorted(by_label.keys()):
        print(f"Label {lbl}: {len(by_label[lbl])} samples")
    
    train_rows: List[List[str]] = []
    val_rows: List[List[str]] = []
    for lbl in sorted(by_label.keys()):
        rows = by_label[lbl]
        random.shuffle(rows)
        if len(rows) <= MIN_CLASS_SAMPLES:
            print(f"[WARN] Label {lbl} has only {len(rows)} samples, skipping from train/val split.")
            continue
        # Carve out stratified test fraction first
        if test_fraction > 0.0:
            n_test = max(1, int(len(rows) * test_fraction))
            test_rows.extend(rows[:n_test])
            rows = rows[n_test:]
        n_val = int(len(rows) * val_fraction)  # 0 when val_fraction=0.0
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])
    if test_fraction > 0.0:
        print(f"Index splitting: fraction {test_fraction:.0%} → test set ({len(test_rows)} samples).")
    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def load_index_csv(path: str) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader) # Skip header
        for r in reader:
            rows.append(r)
    return rows


def compute_class_weights(rows: List[List[str]], num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    for r in rows:
        counts[int(r[1])] += 1
    counts = counts + 1  # Laplace to avoid zero
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_individual_expert(
    expert: LinearProbe,
    train_loader: DataLoader,
    val_loader: DataLoader,
    expert_idx: int,
    expert_name: str,
    device: torch.device,
    num_classes: int,
    class_weights: torch.Tensor,
    output_dir: str,
    id2name: Dict[int, str],
) -> LinearProbe:
    """Train a single expert head independently.
    
    Returns the trained expert with the best validation accuracy.
    """
    print(f"\n{'='*60}")
    print(f"Stage 1: Training Expert {expert_idx} ({expert_name})")
    print(f"{'='*60}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(expert.parameters(), lr=STAGE1_LEARNING_RATE, weight_decay=STAGE1_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS, eta_min=1e-6)
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(1, STAGE1_EPOCHS + 1):
        # Training
        expert.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        t0 = time.time()
        
        for x, y in train_loader:
            y = y.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            logits = expert(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += accuracy(logits.detach(), y)
            n_batches += 1
        
        epoch_loss /= max(1, n_batches)
        epoch_acc /= max(1, n_batches)
        
        # Validation
        expert.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val_batches = 0
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                y = y.to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)
                
                logits = expert(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                val_acc += accuracy(logits, y)
                n_val_batches += 1
                
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        val_loss /= max(1, n_val_batches)
        val_acc /= max(1, n_val_batches)
        
        scheduler.step()

        dt = time.time() - t0
        print(f"  Epoch {epoch:02d}/{STAGE1_EPOCHS} - train_loss {epoch_loss:.4f} acc {epoch_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} | lr {scheduler.get_last_lr()[0]:.2e} ({dt:.1f}s)")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in expert.state_dict().items()}
    
    expert.load_state_dict(best_state)
    
    expert_save_path = os.path.join(output_dir, f"expert_{expert_idx}_{expert_name}_stage1.pt")
    torch.save({
        'state_dict': best_state,
        'val_acc': best_val_acc,
        'embedding_dim': expert.embedding_dim,
        'num_classes': num_classes,
    }, expert_save_path)
    print(f"  Saved to {expert_save_path}")
    
    return expert


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train model ensembles")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Base directory for training outputs.")
    parser.add_argument(
        "--exclude_sources",
        nargs="*",
        default=[],
        help="Source names from index.csv to exclude from train/val/test splitting, e.g. BCSS.",
    )
    parser.add_argument(
        "--test_organs",
        nargs="*",
        default=None,
        help="Override the organ holdout list used for test splitting. Pass the flag with no values to disable organ holdout.",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=None,
        help="Override the stratified test fraction.",
    )
    parser.add_argument(
        "--test_organ",
        type=str,
        default=None,
        help="Override the single held-out organ selected for evaluation.",
    )
    args = parser.parse_args()

    set_seed(SEED)

    expert_configs = MODEL_NAMES
    expert_configs = [sorted(expert) for expert in expert_configs]
    expert_configs.sort()

    model_dir_name = ",".join(["+".join(expert) for expert in expert_configs]) if len(expert_configs) > 1 else "+".join(expert_configs[0])
    print(f"\n{'='*60}")
    for i, expert in enumerate(expert_configs):
        print(f"  Expert {i}: {' + '.join(expert)}")
    print(f"Mode: learned gating weights")
    print(f"{'='*60}\n")
    
    is_single_expert = len(expert_configs) == 1
    model_type_suffix = f"_{MODEL_TYPE}" if is_single_expert and MODEL_TYPE != "mlp" else ""
    model_output_dir = os.path.join(args.output_dir, model_dir_name + model_type_suffix)
    ensemble_ablation_flags = {
        "logit_mlp": USE_LOGIT_MLP_ABLATION,
        "gated_average": USE_GATED_AVERAGE_ABLATION,
        "confidence_weighted": USE_CONFIDENCE_WEIGHTED_ABLATION,
        "organ_ablation": USE_ORGAN_EMBEDDING_ABLATION,
    }
    active_ablations = [name for name, enabled in ensemble_ablation_flags.items() if enabled]
    if not is_single_expert and len(active_ablations) > 1:
        raise ValueError(f"Only one ensemble ablation can be active at a time, got: {active_ablations}")
    if USE_LOGIT_MLP_ABLATION and not is_single_expert:
        model_output_dir += "_mlp_ablation"
    if USE_GATED_AVERAGE_ABLATION and not is_single_expert:
        model_output_dir += "_gated_average_ablation"
    if USE_CONFIDENCE_WEIGHTED_ABLATION and not is_single_expert:
        model_output_dir += "_confidence_weighted_ablation"
    if USE_ORGAN_EMBEDDING_ABLATION and not is_single_expert:
        model_output_dir += "_organ_ablation"
    if USE_ONE_STAGE_ABLATION and not is_single_expert:
        model_output_dir += "_one_stage_ablation"
    if not is_single_expert and not USE_ALTERNATING:
        model_output_dir += "_joint_stage_2"
    ensure_dir(model_output_dir)

    print("torch.version.cuda:", torch.version.cuda)
    print("cudnn available:", torch.backends.cudnn.is_available())
    print("cudnn version:", torch.backends.cudnn.version())

    # Step 1: Prepare tiles + index if not present
    if not os.path.exists(INDEX_CSV):
        print("Index not found — starting tiling and index creation...")
        build_tiles_and_index()
    else:
        print(f"Found existing index at {INDEX_CSV}. Skipping tiling.")

    id2name = UNIFIED_ID_TO_NAME
    num_classes = len(UNIFIED_LABELS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_test_organs = VAL_ORGANS if args.test_organs is None else args.test_organs
    effective_test_fraction = TEST_FRACTION if args.test_fraction is None else args.test_fraction
    effective_test_organ = TEST_ORGAN if args.test_organ is None else args.test_organ

    # Step 4: Load index & split
    all_rows = load_index_csv(INDEX_CSV)
    if len(all_rows) == 0:
        raise RuntimeError("Index CSV has zero rows. Check data + majority threshold.")
    split_rows, excluded_rows = filter_rows_by_sources(all_rows, args.exclude_sources)
    if len(split_rows) == 0:
        raise RuntimeError(f"All rows were excluded by --exclude_sources={args.exclude_sources}.")
    if excluded_rows:
        excluded_counts = Counter(row[3] for row in excluded_rows)
        print(f"Excluded {len(excluded_rows)} rows by source: {dict(excluded_counts)}")

    train_rows, val_rows, test_rows = split_index(
        split_rows,
        VAL_FRACTION,
        test_organs=effective_test_organs,
        test_fraction=effective_test_fraction,
    )
    has_val  = len(val_rows)  > 0
    has_test = len(test_rows) > 0

    # Optional: Subsample training set for faster feedback
    if TRAIN_SUBSET_RATIO < 1.0:
        k = int(len(train_rows) * TRAIN_SUBSET_RATIO)
        print(f"Subsampling training set: Using {k} samples ({TRAIN_SUBSET_RATIO*100:.1f}%) of available {len(train_rows)}")
        random.shuffle(train_rows)
        train_rows = train_rows[:k]

    print(f"Train tiles: {len(train_rows)} | Val tiles: {len(val_rows)} | Test tiles: {len(test_rows)}")

    # Step 5: Load backbones and compute embeddings for each expert
    all_expert_embeddings = []  # List of embeddings for each expert: [expert][samples, features]
    expert_embedding_dims = []

    for expert_idx, expert_models in enumerate(expert_configs):
        print(f"\n{'='*60}")
        print(f"Processing Expert {expert_idx}: {' + '.join(expert_models)}")
        print(f"{'='*60}")
        
        expert_embeddings_list = []
        expert_total_dim = 0

        
        for model_name in expert_models:
            print(f"\nLoading backbone: {model_name}")
            
            # Load backbone based on model_name
            if model_name.lower() == "virchow2":
                backbone = timm.create_model(
                    "hf-hub:paige-ai/Virchow2",
                    pretrained=True,
                    mlp_layer=SwiGLUPacked,
                    act_layer=torch.nn.SiLU,
                )
                cfg = resolve_data_config(backbone.pretrained_cfg, model=backbone)
                transform = create_transform(**cfg)
            elif model_name.lower() == "uni2":
                timm_kwargs = {
                    'img_size': 224, 
                    'patch_size': 14, 
                    'depth': 24,
                    'num_heads': 24,
                    'init_values': 1e-5, 
                    'embed_dim': 1536,
                    'mlp_ratio': 2.66667*2,
                    'num_classes': 0, 
                    'no_embed_class': True,
                    'mlp_layer': timm.layers.SwiGLUPacked, 
                    'act_layer': torch.nn.SiLU, 
                    'reg_tokens': 8, 
                    'dynamic_img_size': True
                }
                backbone = timm.create_model(
                    "hf-hub:MahmoodLab/UNI2-h",
                    pretrained=True,
                    **timm_kwargs
                )
                cfg = resolve_data_config(backbone.pretrained_cfg, model=backbone)
                transform = create_transform(**cfg)
            elif model_name.lower() == "conch":
                backbone, conch_preprocess = create_model_from_pretrained(
                    'conch_ViT-B-16', 
                    "hf_hub:MahmoodLab/conch"
                )
                transform = conch_preprocess
            elif model_name.lower() == "hibou-l":
                processor = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
                backbone = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
                # Use the processor as the transform
                transform = lambda imgs: processor(images=imgs, return_tensors="pt")["pixel_values"].squeeze(0)
            elif model_name.lower() == "h-optimus-0":
                backbone = timm.create_model(
                    "hf_hub:bioptimus/H-optimus-0",
                    pretrained=True,
                    init_values=1e-5, 
                    dynamic_img_size=False
                )
                transform = transforms.Compose([
                    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.707223, 0.578729, 0.703617), 
                        std=(0.211883, 0.230117, 0.177517)
                    ),
                ])
                
            elif model_name.lower() == "h-optimus-1":
                backbone = timm.create_model(
                    "hf_hub:bioptimus/H-optimus-1",
                    pretrained=True,
                    init_values=1e-5, 
                    dynamic_img_size=False
                )
                transform = transforms.Compose([
                    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.707223, 0.578729, 0.703617), 
                        std=(0.211883, 0.230117, 0.177517)
                    ),
                ])
            else:
                raise ValueError(f"Unsupported model name: {model_name}.")
            
            backbone.eval()
            for p in backbone.parameters():
                p.requires_grad = False
            backbone.to(device)
            print(f"Loaded {model_name} on device {device}")

            # Compute embeddings for entire dataset
            all_emb, emb_dim = precompute_embeddings([[row[0], row[1]] for row in all_rows], backbone, transform, device, model_name)
            
            expert_embeddings_list.append(all_emb["X"])
            expert_total_dim += emb_dim
            
            print(f"{model_name} embedding dim: {emb_dim}")
            
            # Free GPU memory
            del backbone
            torch.cuda.empty_cache()
        
        # Concatenate embeddings for this expert
        expert_X = torch.cat(expert_embeddings_list, dim=1)
        
        all_expert_embeddings.append(expert_X)
        expert_embedding_dims.append(expert_total_dim)
        
        print(f"Expert {expert_idx} total embedding dim: {expert_total_dim}")
    
    # Get labels from the full dataset embeddings
    embeddings_subdir = os.path.join(EMBEDDINGS_DIR, expert_configs[0][0])
    all_labels = torch.load(os.path.join(embeddings_subdir, "all.pt"))["y"]
    
    # Split embeddings into train/val based on the index split
    # Create mapping from paths to indices in all_rows
    train_paths_set = set(r[0] for r in train_rows)
    val_paths_set = set(r[0] for r in val_rows)
    all_paths = [r[0] for r in all_rows]
    
    train_indices = [i for i, path in enumerate(all_paths) if path in train_paths_set]
    val_indices = [i for i, path in enumerate(all_paths) if path in val_paths_set]
    
    # Split labels
    train_labels = all_labels[train_indices]
    val_labels = all_labels[val_indices]

    # Split embeddings for each expert
    all_expert_train_embeddings = [expert_X[train_indices] for expert_X in all_expert_embeddings]
    all_expert_val_embeddings   = [expert_X[val_indices]   for expert_X in all_expert_embeddings]

    # Split test embeddings (populated only when VAL_ORGAN is set)
    test_paths_set = set(r[0] for r in test_rows)
    test_indices   = [i for i, path in enumerate(all_paths) if path in test_paths_set]
    test_labels    = all_labels[test_indices] if test_indices else torch.zeros(0, dtype=torch.long)
    all_expert_test_embeddings = [expert_X[test_indices] if test_indices else expert_X[:0]
                                  for expert_X in all_expert_embeddings]

    # Loss with class weighting (to mitigate imbalance)
    class_weights = compute_class_weights(train_rows, num_classes).to(device)
    
    # Extract organ indices for all data (if using organ embedding ablation)
    all_organs = None
    num_organs = 0
    if USE_ORGAN_EMBEDDING_ABLATION:
        organ2id = {org: i for i, org in enumerate(ORGANS)}
        num_organs = len(ORGANS)
        
        # We need to get organ for each sample in all_rows order.
        # all_rows[i][5] is organ string
        all_organs = []
        for r in all_rows:
            org_str = r[5].lower().strip()
            if org_str not in organ2id:
                raise ValueError("Organ not found")
            else:
                idx = organ2id[org_str]
            all_organs.append(idx)
        all_organs = torch.tensor(all_organs, dtype=torch.long)
        
        # Split organs
        train_organs = all_organs[train_indices]
        val_organs   = all_organs[val_indices]
        test_organs  = all_organs[test_indices] if test_indices else torch.zeros(0, dtype=torch.long)

    # Check if we're training an ensemble (multiple experts)
    is_ensemble = len(expert_configs) > 1
    
    if is_ensemble:
        print(f"\n{'='*80}")
        print(f"ENSEMBLE MODE: Cross-Attention with {'Alternating' if USE_ALTERNATING else 'Joint'} Training")
        print(f"  Training: {'one expert at a time (alternating)' if USE_ALTERNATING else 'all experts jointly (ablation)'}")
        print(f"{'='*80}")
        
        # Create ensemble datasets
        train_emb_dict = {"X": all_expert_train_embeddings, "y": train_labels}
        val_emb_dict = {"X": all_expert_val_embeddings, "y": val_labels}
        
        if USE_ORGAN_EMBEDDING_ABLATION:
            train_emb_dict["organ"] = train_organs
            val_emb_dict["organ"] = val_organs

        train_ds = MultiEmbeddingDataset(train_emb_dict)
        val_ds = MultiEmbeddingDataset(val_emb_dict)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        # ============================================================
        # TRAINING: Alternating per-expert training
        # ============================================================
        print(f"\n{'='*60}")
        if USE_LOGIT_MLP_ABLATION:
            print(f"Training LogitMLPEnsemble (Ablation)")
        elif USE_GATED_AVERAGE_ABLATION:
            print(f"Training GatedAverageEnsemble (Ablation)")
        elif USE_CONFIDENCE_WEIGHTED_ABLATION:
            print(f"Training ConfidenceWeightedEnsemble (Ablation)")
        elif USE_ORGAN_EMBEDDING_ABLATION:
            print(f"Training OrganCrossAttentionEnsemble (Ablation)")
        else:
            print(f"Training CrossAttentionEnsemble")
        print(f"{'='*60}")
        
        # Create Ensemble model
        if USE_LOGIT_MLP_ABLATION:
            model = LogitMLPEnsemble(
                num_classes=num_classes,
                expert_embedding_dims=expert_embedding_dims,
                hidden_dim=HIDDEN_DIM,
            ).to(device)
            print(f"Created LogitMLPEnsemble with {len(expert_embedding_dims)} experts")
        elif USE_GATED_AVERAGE_ABLATION:
            model = GatedAverageEnsemble(
                num_classes=num_classes,
                expert_embedding_dims=expert_embedding_dims,
                hidden_dim=HIDDEN_DIM,
            ).to(device)
            print(f"Created GatedAverageEnsemble with {len(expert_embedding_dims)} experts")
        elif USE_CONFIDENCE_WEIGHTED_ABLATION:
            model = ConfidenceWeightedEnsemble(
                num_classes=num_classes,
                expert_embedding_dims=expert_embedding_dims,
                hidden_dim=HIDDEN_DIM,
            ).to(device)
            print(f"Created ConfidenceWeightedEnsemble with {len(expert_embedding_dims)} experts")
        elif USE_ORGAN_EMBEDDING_ABLATION:
            model = OrganCrossAttentionEnsemble(
                num_classes=num_classes,
                expert_embedding_dims=expert_embedding_dims,
                num_organs=num_organs,
                organ_embedding_dim=32,
                hidden_dim=HIDDEN_DIM,
            ).to(device)
            print(f"Created OrganCrossAttentionEnsemble with {len(expert_embedding_dims)} experts") 
        else:
            model = CrossAttentionEnsemble(
                num_classes=num_classes,
                expert_embedding_dims=expert_embedding_dims,
                hidden_dim=HIDDEN_DIM,
            ).to(device)
            print(f"Created CrossAttentionEnsemble with {len(expert_embedding_dims)} experts")

        print(f"  Expert dims: {expert_embedding_dims}")
        num_experts = model.num_experts

        def _freeze_expert(idx, freeze=True):
            """Freeze or unfreeze a single expert's encoder + classifier."""
            rg = not freeze
            for p in model.encoders[idx].parameters():
                p.requires_grad = rg
            for p in model.classifiers[idx].parameters():
                p.requires_grad = rg

        if not USE_ONE_STAGE_ABLATION:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

            stage1_params = list(model.trainable_params())
            optimizer = optim.AdamW(stage1_params, lr=STAGE1_LEARNING_RATE, weight_decay=STAGE1_WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, STAGE1_EPOCHS), eta_min=1e-6
            )

            best_val_acc_stage1 = -1.0
            best_state_stage1 = {k: v.clone() for k, v in model.state_dict().items()}  # fallback = initial model

            _skip_stage1 = STAGE1_CHECKPOINT is not None
            if _skip_stage1:
                print(f"  Skipping Stage 1 — loading checkpoint: {STAGE1_CHECKPOINT}")
                _s1_ckpt = torch.load(STAGE1_CHECKPOINT, map_location=device)
                missing, unexpected = model.load_state_dict(_s1_ckpt["model_state"], strict=False)
                if unexpected:
                    raise RuntimeError(f"Unexpected keys in Stage 1 checkpoint: {unexpected}")
                if missing:
                    print(f"  Keys absent in checkpoint (will use fresh init): {missing}")
                best_val_acc_stage1 = _s1_ckpt.get("val_acc", 0.0)
                best_state_stage1 = {k: v.clone() for k, v in model.state_dict().items()}
                best_val_acc_ensemble = best_val_acc_stage1
                print(f"  Loaded. val_acc={best_val_acc_stage1:.4f}")
            
            def _validate_stage1():
                """Validate using mean of per-expert logits. Returns zeros when val set is empty."""
                if not has_val:
                    return 0.0, 0.0, 0.0, 0.0
                model.eval()
                val_loss = 0.0
                val_acc = 0.0
                n_val_batches = 0
                all_preds_v, all_labels_v = [], []
                with torch.no_grad():
                    for batch_data in val_loader:
                        if USE_ORGAN_EMBEDDING_ABLATION:
                            x, y, organ = batch_data
                            organ = organ.to(device, non_blocking=True)
                        else:
                            x, y = batch_data
                        
                        y = y.to(device, non_blocking=True)
                        x = [xi.to(device, non_blocking=True) for xi in x]
                        
                        if USE_ORGAN_EMBEDDING_ABLATION:
                            expert_logits = model.forward_all_experts(x, organ_ids=organ)
                        else:
                            expert_logits = model.forward_all_experts(x)
                        logits = torch.stack(expert_logits, dim=0).mean(dim=0)
                        loss = criterion(logits, y)
                        val_loss += loss.item()
                        val_acc += accuracy(logits, y)
                        n_val_batches += 1
                        all_preds_v.extend(logits.argmax(dim=1).cpu().numpy())
                        all_labels_v.extend(y.cpu().numpy())
                bal_acc = balanced_accuracy_score(all_labels_v, all_preds_v)
                f1 = f1_score(all_labels_v, all_preds_v, average='weighted', zero_division=0)
                return (val_loss / max(1, n_val_batches),
                        val_acc / max(1, n_val_batches),
                        bal_acc, f1)
            
            # --- Stage 1 Training ---
            mode_str = "alt" if USE_ALTERNATING else "joint"
            total_epochs = STAGE1_EPOCHS
            if not _skip_stage1:
                print(f"\n  Stage 1: {STAGE1_EPOCHS} epochs, mode={mode_str}")
            global_epoch = STAGE1_EPOCHS if _skip_stage1 else 0
            for epoch in (range(1, STAGE1_EPOCHS + 1) if not _skip_stage1 else []):
                global_epoch += 1
                model.train()
                epoch_losses = [0.0] * num_experts
                epoch_accs = [0.0] * num_experts
                n_batches = 0
                t0 = time.time()
                
                for batch_data in train_loader:
                    if USE_ORGAN_EMBEDDING_ABLATION:
                        x, y, organ = batch_data
                        organ = organ.to(device, non_blocking=True)
                    else:
                        x, y = batch_data
                    
                    y = y.to(device, non_blocking=True)
                    x = [xi.to(device, non_blocking=True) for xi in x]
                    
                    if USE_ALTERNATING:
                        # Train each expert individually, freezing the others
                        for ei in range(num_experts):
                            for ej in range(num_experts):
                                _freeze_expert(ej, freeze=(ej != ei))
                            optimizer.zero_grad(set_to_none=True)
                            if USE_ORGAN_EMBEDDING_ABLATION:
                                logits = model.forward_single_expert(x, ei, organ_ids=organ)
                            else:
                                logits = model.forward_single_expert(x, ei)
                            loss = criterion(logits, y)
                            loss.backward()
                            optimizer.step()
                            epoch_losses[ei] += loss.item()
                            epoch_accs[ei] += accuracy(logits.detach(), y)
                        # Unfreeze all experts for next batch
                        for ej in range(num_experts):
                            _freeze_expert(ej, freeze=False)
                    else:
                        # Joint: train all experts simultaneously with mean loss
                        for ej in range(num_experts):
                            _freeze_expert(ej, freeze=False)
                        optimizer.zero_grad(set_to_none=True)
                        if USE_ORGAN_EMBEDDING_ABLATION:
                            expert_logits = model.forward_all_experts(x, organ_ids=organ)
                        else:
                            expert_logits = model.forward_all_experts(x)
                        loss = sum(criterion(lg, y) for lg in expert_logits) / num_experts
                        loss.backward()
                        optimizer.step()
                        for ei, lg in enumerate(expert_logits):
                            epoch_losses[ei] += loss.item()
                            epoch_accs[ei] += accuracy(lg.detach(), y)
                    n_batches += 1
                
                # Validation
                val_loss, val_acc, bal_acc, f1 = _validate_stage1()
                scheduler.step()
                dt = time.time() - t0
                
                loss_strs = " | ".join([f"E{i}={epoch_losses[i]/max(1,n_batches):.4f}" for i in range(num_experts)])
                acc_strs = " | ".join([f"E{i}={epoch_accs[i]/max(1,n_batches):.4f}" for i in range(num_experts)])
                print(f"  Epoch {global_epoch:02d}/{total_epochs} [S1|{mode_str}] - losses [{loss_strs}] accs [{acc_strs}] | "
                    f"val_loss {val_loss:.4f} acc {val_acc:.4f} bal_acc {bal_acc:.4f} f1 {f1:.4f} | lr {scheduler.get_last_lr()[0]:.2e} ({dt:.1f}s)")
                
                if val_acc > best_val_acc_stage1 or not has_val:
                    best_val_acc_stage1 = val_acc
                    best_state_stage1 = {k: v.clone() for k, v in model.state_dict().items()}
            
            if not _skip_stage1:
                # Restore best Stage 1 weights
                model.load_state_dict(best_state_stage1)
                best_val_acc_ensemble = best_val_acc_stage1
                print(f"\n  Stage 1 complete. Best val acc: {best_val_acc_stage1:.4f}")

                # Save Stage 1 checkpoint independently
                stage1_ckpt_path = os.path.join(model_output_dir, "stage1_linear_probe.pt")
                torch.save({
                    "model_state": best_state_stage1,
                    "num_classes": num_classes,
                    "id2name": id2name,
                    "expert_models": expert_configs,
                    "expert_embedding_dims": expert_embedding_dims,
                    "is_ensemble": is_ensemble,
                    "ensemble_method": get_ensemble_method_name(),
                    "hidden_dim": HIDDEN_DIM,
                    "stage": 1,
                    "val_acc": best_val_acc_stage1,
                    "excluded_sources": args.exclude_sources,
                }, stage1_ckpt_path)
                print(f"  Saved Stage 1 checkpoint to {stage1_ckpt_path}")

            print(f"\nTraining complete. Best val acc: {best_val_acc_ensemble:.4f}")
            
            # Final evaluation pass — collect predictions for CSV / confusion matrix
            model.eval()
            val_paths = [r[0] for r in val_rows]
            all_preds, all_labels = [], []
            per_sample_preds, per_sample_probs = [], []
            eval_criterion = nn.CrossEntropyLoss()
            with torch.no_grad():
                for batch_data in val_loader:
                    if USE_ORGAN_EMBEDDING_ABLATION:
                        x, y, organ = batch_data
                        organ = organ.to(device, non_blocking=True)
                    else:
                        x, y = batch_data
                    
                    y = y.to(device, non_blocking=True)
                    x = [xi.to(device, non_blocking=True) for xi in x]
                    
                    if USE_ORGAN_EMBEDDING_ABLATION:
                        logits = model(x, organ_ids=organ)
                    else:
                        logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
                    per_sample_preds.extend(preds.cpu().numpy().tolist())
                    per_sample_probs.extend(probs.cpu().numpy().tolist())
            
            # Save best checkpoint
            best_ckpt_path = os.path.join(model_output_dir, "best_linear_probe.pt")
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "id2name": id2name,
                "expert_models": expert_configs,
                "expert_embedding_dims": expert_embedding_dims,
                "is_ensemble": is_ensemble,
                "ensemble_method": get_ensemble_method_name(),
                "hidden_dim": HIDDEN_DIM,
                "excluded_sources": args.exclude_sources,
            }, best_ckpt_path)
            print(f"Saved best model to {best_ckpt_path}")
            
            # Save final val outputs CSV
            out_csv = os.path.join(model_output_dir, "val_outputs_final.csv")
            with open(out_csv, "w", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["path", "true_label", "pred_label", "top1_prob", "probs_json"])
                for (path_row, true_lbl, pred_lbl, probs_row) in zip(val_paths, all_labels, per_sample_preds, per_sample_probs):
                    top1 = max(probs_row)
                    writer.writerow([path_row, int(true_lbl), int(pred_lbl), float(top1), json.dumps(probs_row)])
            print(f"Saved validation outputs to {out_csv}")
            
            best_val_acc = best_val_acc_ensemble
            history = []
            log_json_path = os.path.join(model_output_dir, "training_log.json")
            with open(log_json_path, "w") as f:
                json.dump({"best_val_acc": best_val_acc}, f, indent=2)

        # --- Stage 2: fine-tune with HiddenCrossAttention ---
        # Differential LR: CA params learn fast, existing params fine-tune slowly.
        if USE_LOGIT_MLP_ABLATION:
            ca_params = list(model.mlp.parameters())
        elif USE_GATED_AVERAGE_ABLATION:
            ca_params = list(model.gate.parameters())
        elif USE_CONFIDENCE_WEIGHTED_ABLATION:
            ca_params = [model.confidence_temperature]
        elif USE_ORGAN_EMBEDDING_ABLATION:
            ca_params = list(model.cross_attention.parameters())
        else:
            ca_params = list(model.cross_attention.parameters())

        existing_params = list(model.trainable_params())  # encoders + classifiers + temperatures
        optimizer = optim.AdamW([
            {"params": existing_params, "lr": STAGE2_LR_EXISTING if not USE_ONE_STAGE_ABLATION else STAGE1_LEARNING_RATE},
            {"params": ca_params,       "lr": STAGE2_LEARNING_RATE},
        ], weight_decay=STAGE1_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, STAGE2_EPOCHS if not USE_ONE_STAGE_ABLATION else ONE_STAGE_EPOCHS), eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        MAX_EPOCHS = STAGE2_EPOCHS if not USE_ONE_STAGE_ABLATION else ONE_STAGE_EPOCHS
        
    else:
        # Single expert mode
        print(f"\n{'='*60}")
        print(f"SINGLE EXPERT MODE ({MODEL_TYPE.upper()}): Training linear head")
        print(f"{'='*60}")

        train_emb_dict = {"X": all_expert_train_embeddings[0], "y": train_labels}
        val_emb_dict = {"X": all_expert_val_embeddings[0], "y": val_labels}

        train_ds = EmbeddingDataset(train_emb_dict)
        val_ds = EmbeddingDataset(val_emb_dict)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        if MODEL_TYPE == "logreg":
            # Multinomial logistic regression: single linear layer, no hidden representation
            model = nn.Linear(expert_embedding_dims[0], num_classes).to(device)
            criterion = nn.CrossEntropyLoss()  # no label smoothing — pure baseline
            optimizer = optim.Adam(model.parameters(), lr=STAGE1_LEARNING_RATE)
            print(f"  Model: nn.Linear({expert_embedding_dims[0]}, {num_classes}) — logistic regression")
        else:
            # MLP: embedding → Linear → GELU → Linear
            model = LinearProbe(num_classes=num_classes, embedding_dim=expert_embedding_dims[0]).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=STAGE1_LEARNING_RATE, weight_decay=STAGE1_WEIGHT_DECAY)
            print(f"  Model: LinearProbe (MLP, hidden_dim={HIDDEN_DIM})")

        MAX_EPOCHS = STAGE1_EPOCHS if not USE_ONE_STAGE_ABLATION else ONE_STAGE_EPOCHS
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
        
    # Training loop (Stage 2 for ensemble, or full training for single expert)
    best_val_acc = 0.0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_data in train_loader:
            if USE_ORGAN_EMBEDDING_ABLATION:
                x, y, organ = batch_data
                organ = organ.to(device, non_blocking=True)
            else:
                x, y = batch_data

            y = y.to(device, non_blocking=True)

            if is_ensemble:
                x = [x_i.to(device, non_blocking=True) for x_i in x]
                if USE_ALTERNATING:
                    # Stage 2 alternating: train one expert at a time (CA params always update)
                    for ei in range(num_experts):
                        for ej in range(num_experts):
                            _freeze_expert(ej, freeze=(ej != ei))
                        optimizer.zero_grad(set_to_none=True)
                        if USE_ORGAN_EMBEDDING_ABLATION:
                            logits = model(x, organ_ids=organ)
                        else:
                            logits = model(x)  # full forward — cross-attention active
                        loss = criterion(logits, y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        epoch_acc += accuracy(logits.detach(), y)
                    for ej in range(num_experts):
                        _freeze_expert(ej, freeze=False)
                else:
                    # Stage 2 joint: all experts + CA trained together
                    for ej in range(num_experts):
                        _freeze_expert(ej, freeze=False)
                    optimizer.zero_grad(set_to_none=True)
                    if USE_ORGAN_EMBEDDING_ABLATION:
                        logits = model(x, organ_ids=organ)
                    else:
                        logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    epoch_acc += accuracy(logits.detach(), y)
            else:
                x = x.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += accuracy(logits.detach(), y)
            n_batches += 1

        epoch_loss /= max(1, n_batches)
        epoch_acc /= max(1, n_batches)
        scheduler.step()

        # Validation
        val_loss, val_acc, bal_acc, f1 = 0.0, 0.0, 0.0, 0.0
        all_preds, all_labels = [], []
        val_paths = [r[0] for r in val_rows]
        per_sample_preds, per_sample_probs = [], []

        if has_val:
            model.eval()
            n_val_batches = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    if USE_ORGAN_EMBEDDING_ABLATION:
                        x, y, organ = batch_data
                        organ = organ.to(device, non_blocking=True)
                    else:
                        x, y = batch_data
                    
                    y = y.to(device, non_blocking=True)
                    # Handle ensemble vs single expert
                    if is_ensemble:
                        x = [x_i.to(device, non_blocking=True) for x_i in x]
                    else:
                        x = x.to(device, non_blocking=True)
                        
                    if USE_ORGAN_EMBEDDING_ABLATION and is_ensemble:
                        logits = model(x, organ_ids=organ)
                    else:
                        logits = model(x)
                        
                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)
                    loss = criterion(logits, y)
                    val_loss += loss.item()
                    val_acc += accuracy(logits, y)
                    n_val_batches += 1
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
                    per_sample_preds.extend(preds.cpu().numpy().tolist())
                    per_sample_probs.extend(probs.cpu().numpy().tolist())
            val_loss /= max(1, n_val_batches)
            val_acc  /= max(1, n_val_batches)
            bal_acc = balanced_accuracy_score(all_labels, all_preds)
            f1      = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        dt = time.time() - t0
        log_row = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_loss":   val_loss   if has_val else None,
            "val_acc":    val_acc    if has_val else None,
            "bal_acc":    bal_acc    if has_val else None,
            "f1_weighted": f1       if has_val else None,
            "seconds": dt,
        }
        history.append(log_row)
        if has_val:
            print(
                f"Epoch {epoch:02d}/{MAX_EPOCHS} - train_loss {epoch_loss:.4f} acc {epoch_acc:.4f} | "
                f"val_loss {val_loss:.4f} acc {val_acc:.4f} bal_acc {bal_acc:.4f} f1 {f1:.4f} | lr {scheduler.get_last_lr()[0]:.2e} ({dt:.1f}s)"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{MAX_EPOCHS} - train_loss {epoch_loss:.4f} acc {epoch_acc:.4f} | "
                f"(no validation) | lr {scheduler.get_last_lr()[0]:.2e} ({dt:.1f}s)"
            )

        # Save checkpoint: best-by-val when has_val, otherwise every epoch (last epoch wins)
        if val_acc > best_val_acc or not has_val:
            best_val_acc = val_acc
            best_ckpt_path = os.path.join(model_output_dir, "best_linear_probe.pt")
            checkpoint = {
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "id2name": id2name,
                "expert_models": expert_configs,
                "expert_embedding_dims": expert_embedding_dims,
                "is_ensemble": is_ensemble,
                "ensemble_method": get_ensemble_method_name() if is_ensemble else "single",
                "hidden_dim": HIDDEN_DIM,
                "excluded_sources": args.exclude_sources,
            }
            if not is_ensemble:
                checkpoint["model_type"] = MODEL_TYPE
            torch.save(checkpoint, best_ckpt_path)
            print(f"  -> Saved new best to {best_ckpt_path}")

        # Save per-epoch validation outputs CSV (skipped when no val)
        if has_val:
            out_csv = os.path.join(model_output_dir, f"val_outputs_epoch_{epoch:02d}.csv")
            # Columns: path, true_label, pred_label, top1_prob, probs_json
            with open(out_csv, "w", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["path", "true_label", "pred_label", "top1_prob", "probs_json"])
                for (path_row, true_lbl, pred_lbl, probs_row) in zip(val_paths, all_labels, per_sample_preds, per_sample_probs):
                    top1 = max(probs_row) if isinstance(probs_row, (list, tuple)) else float(probs_row)
                    writer.writerow([path_row, int(true_lbl), int(pred_lbl), float(top1), json.dumps(probs_row)])
            print(f"Saved validation outputs to {out_csv}")

        # write log each epoch
        log_json_path = os.path.join(model_output_dir, "training_log.json")
        with open(log_json_path, "w") as f:
            json.dump({"history": history, "best_val_acc": best_val_acc}, f, indent=2)
    
    # Compute confusion matrices (only when there is validation data)
    if has_val and len(all_labels) > 0:
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

        cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)

        # Class labels
        disp_labels = [id2name[i] if i in id2name else str(i) for i in range(num_classes)]
        epoch_label = epoch if MAX_EPOCHS > 0 else "Final"

        # ---- RAW (counts) ----
        _, ax = plt.subplots(figsize=(14, 14))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
        disp.plot(xticks_rotation=45, cmap="Blues", values_format="d", ax=ax)
        plt.xticks(rotation=60, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.title(f"Confusion Matrix (Counts) – Epoch {epoch_label}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_dir, f"confusion_matrix.png"), dpi=300)
        plt.close()

        # ---- NORMALIZED (percentages) ----
        _, ax = plt.subplots(figsize=(14, 14))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=disp_labels)
        disp.plot(xticks_rotation=45, cmap="Blues", values_format=".2f", ax=ax, colorbar=True)
        plt.xticks(rotation=60, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.title(f"Confusion Matrix (Normalized) – Epoch {epoch_label}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_dir, f"confusion_matrix_normalized.png"), dpi=300)
        plt.close()

        # ---- Save Text/CSV Version of Confusion Matrix & Class Counts ----
        cm_csv_path = os.path.join(model_output_dir, "confusion_matrix.csv")
        with open(cm_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header: True Label \ Predicted Label
            writer.writerow(["True Label"] + [f"Pred_{label}" for label in disp_labels] + ["Total Samples"])
            
            for i, row in enumerate(cm_normalized):
                true_label_name = disp_labels[i]
                count = int(cm[i].sum())
                # Format row as list of strings
                row_str = [f"{val:.6f}" for val in row]
                writer.writerow([true_label_name] + row_str + [count])
        
        # Save a simple summary text file
        summary_path = os.path.join(model_output_dir, "validation_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Validation Summary - Epoch {epoch_label}\n")
            f.write("="*40 + "\n")
            f.write(f"Total Validation Samples: {len(all_labels)}\n\n")
            f.write(f"{'Class Name':<20} | {'Count':<10} | {'Percentage':<10}\n")
            f.write("-" * 46 + "\n")
            
            total_samples = len(all_labels)
            for i, name in enumerate(disp_labels):
                count = int(cm[i].sum())
                pct = (count / total_samples) * 100 if total_samples > 0 else 0
                f.write(f"{name:<20} | {count:<10} | {pct:6.2f}%\n")
                
        print(f"Saved confusion matrix CSV to {cm_csv_path}")
        print(f"Saved validation summary to {summary_path}")

    print(f"Done. Best val acc: {best_val_acc:.4f}")

    # ---- Test set evaluation (only when VAL_ORGANS was set) ----
    if has_test:
        # Optionally filter to a single organ for evaluation
        if effective_test_organ:
            _eval_organ = effective_test_organ.lower().strip()
            # test_rows only has [path, label]; organ info comes from all_rows via path lookup
            path_to_organ = {r[0]: r[5].lower().strip() for r in all_rows}
            eval_test_rows = [r for r in test_rows if path_to_organ.get(r[0], "") == _eval_organ]
            if not eval_test_rows:
                print(f"[WARN] TEST_ORGAN='{effective_test_organ}' not found in held-out test rows — skipping evaluation.")
                eval_test_rows = []
            _test_organs_str = effective_test_organ
        else:
            eval_test_rows = test_rows
            if effective_test_organs:
                _test_organs_str = ",".join(effective_test_organs)
            else:
                _test_organs_str = f"fraction{int(effective_test_fraction * 100)}pct"

        if not eval_test_rows:
            print(f"No test samples to evaluate.")
        else:
        
            print(f"\n{'='*60}")
            print(f"Evaluating held-out test set: '{_test_organs_str}' ({len(eval_test_rows)} samples)")
            print(f"{'='*60}")


            eval_paths_set = set(r[0] for r in eval_test_rows)
            eval_indices   = [i for i, path in enumerate(all_paths) if path in eval_paths_set]
            test_paths = [all_paths[i] for i in eval_indices]

            # Build eval embeddings/labels from the filtered rows
            path_to_label  = {r[0]: int(r[1]) for r in eval_test_rows}
            # Labels must follow all_paths order (same as eval_indices), not eval_test_rows order
            eval_labels    = torch.tensor([path_to_label[all_paths[i]] for i in eval_indices], dtype=torch.long)
            eval_expert_embeddings = [expert_X[eval_indices] for expert_X in all_expert_embeddings]

            if is_ensemble:
                eval_emb_dict = {"X": eval_expert_embeddings, "y": eval_labels}
                if USE_ORGAN_EMBEDDING_ABLATION:
                    # Filter organs for eval
                    eval_organs = test_organs if effective_test_organ else (all_organs[eval_indices] if eval_indices else torch.zeros(0, dtype=torch.long))
                    # The mapping above might be slightly tricky.
                    # Correct logic:
                    # eval_indices are indices into all samples that match criteria.
                    # all_organs is aligned with all samples.
                    eval_organs = all_organs[eval_indices]
                    eval_emb_dict["organ"] = eval_organs

                eval_ds = MultiEmbeddingDataset(eval_emb_dict)
            else:
                eval_emb_dict = {"X": eval_expert_embeddings[0], "y": eval_labels}
                eval_ds = EmbeddingDataset(eval_emb_dict)

            eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=True)

            model.eval()
            test_preds_list, test_probs_list, test_labels_list = [], [], []
            with torch.no_grad():
                for batch_data in eval_loader:
                    if USE_ORGAN_EMBEDDING_ABLATION:
                        x, y, organ = batch_data
                        organ = organ.to(device, non_blocking=True)
                    else:
                        x, y = batch_data
                    
                    y = y.to(device, non_blocking=True)
                    if is_ensemble:
                        x = [xi.to(device, non_blocking=True) for xi in x]
                    else:
                        x = x.to(device, non_blocking=True)
                    
                    if USE_ORGAN_EMBEDDING_ABLATION and is_ensemble:
                        logits = model(x, organ_ids=organ)
                    else:
                        logits = model(x)
                    
                    probs  = torch.softmax(logits, dim=1)
                    preds  = logits.argmax(dim=1)
                    test_preds_list.extend(preds.cpu().numpy().tolist())
                    test_probs_list.extend(probs.cpu().numpy().tolist())
                    test_labels_list.extend(y.cpu().numpy().tolist())

            if len(test_labels_list) > 0:
                test_acc     = sum(p == l for p, l in zip(test_preds_list, test_labels_list)) / len(test_labels_list)
                test_bal_acc = balanced_accuracy_score(test_labels_list, test_preds_list)
                test_f1      = f1_score(test_labels_list, test_preds_list, average='weighted', zero_division=0)
                print(f"Test ({_test_organs_str}): acc={test_acc:.4f}  bal_acc={test_bal_acc:.4f}  f1={test_f1:.4f}")
            else:
                test_acc = test_bal_acc = test_f1 = 0.0
                print("[WARN] Test set is empty — no metrics computed.")

            # Save test predictions CSV
            test_csv_path = os.path.join(model_output_dir, f"test_outputs_{_test_organs_str}.csv")
            with open(test_csv_path, "w", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["path", "y_true", "y_pred", "top1_prob", "probs_json"])
                for path_row, true_lbl, pred_lbl, probs_row in zip(
                        test_paths, test_labels_list, test_preds_list, test_probs_list):
                    top1 = max(probs_row)
                    writer.writerow([path_row, int(true_lbl), int(pred_lbl), float(top1), json.dumps(probs_row)])
            print(f"Saved test outputs to {test_csv_path}")

            # Append test metrics to the training log
            log_json_path = os.path.join(model_output_dir, "training_log.json")
            if os.path.exists(log_json_path):
                with open(log_json_path) as f:
                    log_data = json.load(f)
            else:
                log_data = {}
            log_data["test_organ"]   = _test_organs_str
            log_data["test_acc"]     = test_acc
            log_data["test_bal_acc"] = test_bal_acc
            log_data["test_f1"]      = test_f1
            with open(log_json_path, "w") as f:
                json.dump(log_data, f, indent=2)

    # ---- Excluded source evaluation (e.g. BCSS excluded from training) ----
    if excluded_rows:
        print(f"\n{'='*60}")
        print(f"Evaluating excluded rows ({len(excluded_rows)} samples)")
        print(f"{'='*60}")

        eval_paths_set = set(r[0] for r in excluded_rows)
        eval_indices = [i for i, path in enumerate(all_paths) if path in eval_paths_set]
        eval_paths = [all_paths[i] for i in eval_indices]

        path_to_label = {r[0]: int(r[1]) for r in excluded_rows}
        path_to_source = {r[0]: r[3] for r in excluded_rows}
        path_to_original_label = {r[0]: r[4] for r in excluded_rows}
        path_to_organ = {r[0]: r[5] for r in excluded_rows}

        eval_labels = torch.tensor([path_to_label[all_paths[i]] for i in eval_indices], dtype=torch.long)
        eval_expert_embeddings = [expert_X[eval_indices] for expert_X in all_expert_embeddings]

        if is_ensemble:
            eval_emb_dict = {"X": eval_expert_embeddings, "y": eval_labels}
            if USE_ORGAN_EMBEDDING_ABLATION:
                eval_emb_dict["organ"] = all_organs[eval_indices]
            eval_ds = MultiEmbeddingDataset(eval_emb_dict)
        else:
            eval_emb_dict = {"X": eval_expert_embeddings[0], "y": eval_labels}
            eval_ds = EmbeddingDataset(eval_emb_dict)

        eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        model.eval()
        excluded_preds_list, excluded_probs_list, excluded_labels_list = [], [], []
        with torch.no_grad():
            for batch_data in eval_loader:
                if USE_ORGAN_EMBEDDING_ABLATION:
                    x, y, organ = batch_data
                    organ = organ.to(device, non_blocking=True)
                else:
                    x, y = batch_data

                y = y.to(device, non_blocking=True)
                if is_ensemble:
                    x = [xi.to(device, non_blocking=True) for xi in x]
                else:
                    x = x.to(device, non_blocking=True)

                if USE_ORGAN_EMBEDDING_ABLATION and is_ensemble:
                    logits = model(x, organ_ids=organ)
                else:
                    logits = model(x)

                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                excluded_preds_list.extend(preds.cpu().numpy().tolist())
                excluded_probs_list.extend(probs.cpu().numpy().tolist())
                excluded_labels_list.extend(y.cpu().numpy().tolist())

        if len(excluded_labels_list) > 0:
            excluded_acc = sum(p == l for p, l in zip(excluded_preds_list, excluded_labels_list)) / len(excluded_labels_list)
            excluded_bal_acc = balanced_accuracy_score(excluded_labels_list, excluded_preds_list)
            excluded_f1 = f1_score(excluded_labels_list, excluded_preds_list, average="weighted", zero_division=0)
            print(
                f"Excluded rows: acc={excluded_acc:.4f}  "
                f"bal_acc={excluded_bal_acc:.4f}  f1={excluded_f1:.4f}"
            )
        else:
            print("[WARN] Excluded rows are empty — no metrics computed.")

        excluded_csv_path = os.path.join(model_output_dir, "excluded_rows_predictions.csv")
        with open(excluded_csv_path, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["path", "y_true", "y_pred", "top1_prob", "probs_json", "source", "original_label", "organ"])
            for path_row, true_lbl, pred_lbl, probs_row in zip(eval_paths, excluded_labels_list, excluded_preds_list, excluded_probs_list):
                writer.writerow([
                    path_row,
                    int(true_lbl),
                    int(pred_lbl),
                    float(max(probs_row)),
                    json.dumps(probs_row),
                    path_to_source[path_row],
                    path_to_original_label[path_row],
                    path_to_organ[path_row],
                ])
        print(f"Saved excluded-row outputs to {excluded_csv_path}")

if __name__ == "__main__":
    main()
