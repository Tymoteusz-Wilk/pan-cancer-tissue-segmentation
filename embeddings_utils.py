from typing import List
import torch
import torch.nn as nn
from data_utils import *
from torch.utils.data import DataLoader

import os
import tqdm

EMBEDDINGS_DIR = os.path.join(UNIFIED_DATA_ROOT, "embeddings")

BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 4)


def compute_virchow2_embeddings(imgs, backbone):
    out = backbone(imgs)    
    class_token = out[:, 0]
    patch_tokens = out[:, 5:]
    pooled = patch_tokens.mean(1)
    return torch.cat([class_token, pooled], dim=-1)  # [B, 2560]


def compute_uni2_embeddings(imgs, backbone):
    return backbone(imgs) #[B, 1536]


def compute_conch_embeddings(imgs, backbone):
    # CONCH uses encode_image method with specific parameters
    # proj_contrast=False, normalize=False gives embeddings suitable for linear probing
    return backbone.encode_image(imgs, proj_contrast=False, normalize=False)  # [B, 512]


def compute_hibou_embeddings(imgs, backbone):
    # Hibou uses transformers DINOv2 architecture
    # Returns last_hidden_state where [:, 0, :] is the CLS token
    outputs = backbone(imgs)
    return outputs.last_hidden_state[:, 0, :]  # [B, 1024] for Hibou-L


def compute_h_optimus_0_embeddings(imgs, backbone):
    # H-optimus-0 returns the CLS token directly from the forward pass
    return backbone(imgs) # [B, 1536]


def compute_model_embeddings(imgs, backbone, model_name):
    if model_name.lower() == "virchow2":
        return compute_virchow2_embeddings(imgs, backbone)
    elif model_name.lower() == "uni2":
        return compute_uni2_embeddings(imgs, backbone)
    elif model_name.lower() == "conch":
        return compute_conch_embeddings(imgs, backbone)
    elif model_name.lower() == "hibou-l":
        return compute_hibou_embeddings(imgs, backbone)
    elif model_name.lower() == "h-optimus-0":
        return compute_h_optimus_0_embeddings(imgs, backbone)
    else:
        raise NotImplementedError(f"Embedding computation not implemented for model {model_name}")


def precompute_embeddings(rows: List[List[str]], backbone: nn.Module, transform, device, model_name):
    """

    Args:
        rows: list of index rows where row[0]=path, row[1]=label (int or str)
        backbone: a timm model instance that returns tokens where class token is at index 0 and patch tokens start at index 5
        transform: preprocessing transform for the images
        device: torch device to use for computation
        model_name: name of the foundational model (e.g. Virchow2, UNI2, etc.)
        
    Returns:
        A dict with keys: 'X' (FloatTensor [N,D]), 'y' (LongTensor [N]), 'paths' (list of str)
        and the embedding dimension (int)
    """
    embeddings_subdir = os.path.join(EMBEDDINGS_DIR, model_name)
    if not os.path.exists(embeddings_subdir):
        os.makedirs(embeddings_subdir, exist_ok=True)

    out_csv = os.path.join(embeddings_subdir, "all.pt")

    X_existing = torch.empty((0,))
    y_existing = torch.empty((0,), dtype=torch.long)

    # If the embeddings file already exists, load and check size
    if os.path.exists(out_csv):
        print(f"Found existing embeddings at {out_csv}, loading...")
        data = torch.load(out_csv, map_location="cpu")
        # Ensure minimal keys exist
        if "X" not in data or "y" not in data:
            raise RuntimeError(f"Embeddings file {out_csv} is missing required keys 'X' or 'y'.")
        
        X_existing = data["X"]
        y_existing = data["y"]

        num_existing = X_existing.shape[0]
        num_total = len(rows)

        if num_existing == num_total:
            print(f"Embeddings count ({num_existing}) matches row count. Returning cached.")
            return data, X_existing.shape[1]

        print(f"Existing embeddings ({num_existing}) < required rows ({num_total}). Computing {num_total - num_existing} new embeddings...")
        rows_to_compute = rows[num_existing:]
        
    else:
        print("No existing embeddings found, computing from scratch...")
        rows_to_compute = rows

    # Compute missing embeddings
    labels: List[int] = []
    embeds: List[torch.Tensor] = []
    
    if len(rows_to_compute) > 0:
        # Create a simple DataLoader over the TileDataset to batch IO and preprocessing
        tile_ds = TileDataset(rows_to_compute, transform)
        loader = DataLoader(tile_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        for batch in tqdm.tqdm(loader, desc=f"Embedding {model_name}"):
            # TileDataset returns (img_tensor, label)
            imgs, ys = batch
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=imgs.is_cuda):
                    emb = compute_model_embeddings(imgs, backbone, model_name)
                    embeds.append(emb.cpu())
                    labels.extend(ys.cpu().numpy().tolist())
                
        X_new = torch.cat(embeds, dim=0)
        y_new = torch.tensor(labels, dtype=torch.long)
    else:
        X_new = torch.empty((0, X_existing.shape[1] if X_existing.numel() > 0 else 0))
        y_new = torch.empty((0,), dtype=torch.long)

    # Concatenate with existing
    if X_existing.numel() > 0:
        if X_new.numel() > 0:
            X_final = torch.cat([X_existing, X_new], dim=0)
            y_final = torch.cat([y_existing, y_new], dim=0)
        else:
            X_final, y_final = X_existing, y_existing
    else:
        X_final, y_final = X_new, y_new

    # Save updated
    out = {"X": X_final, "y": y_final}
    torch.save(out, out_csv)
    print(f"Saved updated embeddings to {out_csv} (N={X_final.shape[0]}, dim={X_final.shape[1]})")
    return out, X_final.shape[1]




