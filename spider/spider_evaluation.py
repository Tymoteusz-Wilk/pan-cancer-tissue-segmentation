import sys
import os
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import timm
from timm.data import resolve_data_config
from timm.layers import SwiGLUPacked
from transformers import AutoImageProcessor, AutoModel
import csv

# Add parent directory to path to import modules from Model Training root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import (
    SPIDER_SKIN_DATA_ROOT, SPIDER_THORAX_DATA_ROOT, 
    SPIDER_BREAST_DATA_ROOT, SPIDER_COLORECTAL_DATA_ROOT,
    SPIDER_SKIN_TO_UNIFIED, SPIDER_THORAX_TO_UNIFIED, 
    SPIDER_BREAST_TO_UNIFIED, SPIDER_COLORECTAL_TO_UNIFIED,
    UNIFIED_NAME_TO_ID, UNIFIED_LABELS, ORGANS,
    DATA_ROOT
)

from models import (
    LinearProbe,
    CrossAttentionEnsemble,
    LogitMLPEnsemble,
    GatedAverageEnsemble,
    ConfidenceWeightedEnsemble,
    OrganCrossAttentionEnsemble,
    HIDDEN_DIM,
)
from embeddings_utils import compute_model_embeddings

from spider_dataset import SPIDERDataset

from conch.open_clip_custom import create_model_from_pretrained

SEED = 100


def get_dataset_info(dataset_name):
    if dataset_name.lower() == 'skin':
        return SPIDER_SKIN_DATA_ROOT, SPIDER_SKIN_TO_UNIFIED, 'skin'
    elif dataset_name.lower() == 'thorax':
        return SPIDER_THORAX_DATA_ROOT, SPIDER_THORAX_TO_UNIFIED, 'thorax'
    elif dataset_name.lower() == 'breast':
        return SPIDER_BREAST_DATA_ROOT, SPIDER_BREAST_TO_UNIFIED, 'breast'
    elif dataset_name.lower() == 'colorectal':
        return SPIDER_COLORECTAL_DATA_ROOT, SPIDER_COLORECTAL_TO_UNIFIED, 'colorectal'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Options: skin, thorax, breast, colorectal")


class SpiderTestDataset(Dataset):
    def __init__(self, spider_dataset, mapping, transform=None):
        self.spider_dataset = spider_dataset
        self.mapping = mapping
        self.transform = transform
        self.indices = []
        
        # Pre-filter valid indices that have a mapping to Unified labels
        print("Filtering dataset for valid labels...")
        for i in range(len(spider_dataset)):
             rec = spider_dataset.records[i]
             cls = rec["class"]
             if cls in mapping:
                 self.indices.append(i)
        print(f"Retained {len(self.indices)} / {len(spider_dataset)} samples.")
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, spider_label_id = self.spider_dataset[real_idx]
        
        # Apply transform if provided
        if self.transform:
            img = self.transform(img)
            
        # Map label
        spider_label_name = self.spider_dataset.label_to_class[spider_label_id]
        unified_label_name = self.mapping[spider_label_name]
        unified_label_id = UNIFIED_NAME_TO_ID[unified_label_name]
        
        return img, unified_label_id


def load_backbone(model_name, device):
    print(f"Loading backbone: {model_name}")
    transform = None
    backbone = None
    
    if model_name.lower() == "virchow2":
        backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        cfg = resolve_data_config(backbone.pretrained_cfg, model=backbone)
        transform = timm.data.transforms_factory.create_transform(**cfg, is_training=False)
        
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
        transform = timm.data.transforms_factory.create_transform(**cfg, is_training=False)
        
    elif model_name.lower() == "conch":
        backbone, conch_preprocess = create_model_from_pretrained(
            'conch_ViT-B-16', 
            "hf_hub:MahmoodLab/conch"
        )
        transform = conch_preprocess
        
    elif model_name.lower() == "hibou-l":
        processor = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
        backbone = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
        transform = lambda imgs: processor(images=imgs, return_tensors="pt")["pixel_values"].squeeze(0)
        
    elif model_name.lower() == "h-optimus-0":
        backbone = timm.create_model(
            "hf_hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5, 
            dynamic_img_size=False
        )
        cfg = resolve_data_config(backbone.pretrained_cfg, model=backbone)
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['mean'], std=cfg['std']),
        ])

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    backbone.eval()
    backbone.to(device)
    return backbone, transform


def get_embedding_dim(model_name):
    if model_name.lower() == "virchow2":
        return 2560
    elif model_name.lower() == "uni2":
        return 1536
    elif model_name.lower() == "conch":
        return 512
    elif model_name.lower() == "hibou-l":
        return 1024
    elif model_name.lower() == "h-optimus-0":
        return 1536
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on SPIDER dataset")
    parser.add_argument("--dataset", type=str, required=True, help="SPIDER dataset to evaluate on (skin, thorax, breast, colorectal)")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing the trained model (e.g., outputs/CONCH,Hibou-L...)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--output_file", type=str, default="evaluation_results.csv", help="Output file for predictions")
    
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 1. Setup Dataset
    data_root, mapping, organ_name = get_dataset_info(args.dataset)
    print(f"Loading SPIDER dataset from {data_root}...")
    
    # Check if data_root exists
    if not os.path.exists(data_root):
        print(f"Error: Data directory {data_root} does not exist.")
        return

    spider_ds = SPIDERDataset(data_root, context_size=1, split="test", transform=None)
    # Note: We pass transform=None here because we deal with transforms per backbone later
    
    # Parse Model Configuration
    model_dir_name = os.path.basename(os.path.normpath(args.model_dir))
    print(f"Model directory name: {model_dir_name}")

    # Strip optional _logreg / _mlp suffix added by training.py for single-expert models
    MODEL_TYPE_SUFFIXES = (
        "_logreg",
        "_mlp_ablation",
        "_gated_average_ablation",
        "_confidence_weighted_ablation",
        "_mlp",
        "_organ_ablation",
    )
    clean_name = model_dir_name
    for suffix in MODEL_TYPE_SUFFIXES:
        if clean_name.endswith(suffix):
            clean_name = clean_name[: -len(suffix)]
            print(f"Stripped '{suffix}' suffix → '{clean_name}'")
            break

    # Parse experts
    expert_configs = []
    if ',' in clean_name:
        parts = clean_name.split(',')
        for p in parts:
            expert_configs.append(p.split('+'))
    else:
        expert_configs.append(clean_name.split('+'))
        
    print(f"Parsed expert configuration: {expert_configs}")
    
    # 3. Compute Embeddings
    # We need to compute embeddings for all backbones involved
    # Identify unique backbones to avoid loading same backbone multiple times if not needed (though training.py doesn't share)
    # But here we should try to be efficient.

    # To avoid OOM and repeated loading, we iterate over unique backbones found in config
    unique_backbones = set()
    for exp in expert_configs:
        for model in exp:
            unique_backbones.add(model)
            
    # We will store embeddings in a dictionary: backbone_name -> Tensor[N, D]
    # To map dataset index to embedding row correctly, we need to iterate the filtered dataset consistently.
    
    
    backbone_embeddings = {}
    test_labels = None
    
    # Setup evaluation cache directory
    evaluation_dir = os.path.join(DATA_ROOT, "Evaluation", args.dataset)
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir, exist_ok=True)
    
    # Compute embeddings for each unique backbone
    for model_name in unique_backbones:
        cache_path = os.path.join(evaluation_dir, f"{model_name}.pt")

        if os.path.exists(cache_path):
            print(f"Loading cached embeddings for {model_name} from {cache_path}")
            cache_data = torch.load(cache_path, map_location="cpu")
            backbone_embeddings[model_name] = cache_data['X']
            
            if test_labels is None:
                test_labels = cache_data['y']
                print(f"Loaded {len(test_labels)} labels from cache.")
        else:
            backbone, transform = load_backbone(model_name, device)
            
            # Create a temporary loader with the specific transform
            # We need to apply transform on-the-fly. SpiderTestDataset can accept a transform.
            ds_for_backbone = SpiderTestDataset(spider_ds, mapping, transform=transform)
            loader = DataLoader(
                ds_for_backbone, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=4
            )
            
            embeddings_list = []
            batch_labels_list = []
            
            print(f"Computing embeddings for {model_name}...")
            with torch.no_grad():
                for imgs, lbls in tqdm(loader):
                    imgs = imgs.to(device)
                    emb = compute_model_embeddings(imgs, backbone, model_name)
                    embeddings_list.append(emb.cpu())
                    
                    if test_labels is None:
                        batch_labels_list.append(lbls)
            
            full_emb = torch.cat(embeddings_list, dim=0)
            backbone_embeddings[model_name] = full_emb
            
            if test_labels is None:
                # If we're computing embeddings, we must have collected labels
                current_labels = torch.cat(batch_labels_list, dim=0)
                test_labels = current_labels 
            else:
                # Use the already loaded labels
                current_labels = test_labels

            print(f"Collected {len(current_labels)} labels.")
            
            # Save to cache
            print(f"Saving embeddings and labels to {cache_path}")
            torch.save({'X': full_emb, 'y': current_labels}, cache_path)
            
            # Free memory
            del backbone
            del transform
            del loader
            del ds_for_backbone
            torch.cuda.empty_cache()
        
    # 4. Construct Expert Inputs
    expert_embeddings_list = [] # List of [N, D_expert]
    expert_dims = []
    
    for config in expert_configs:
        # config is list of model names for this expert (e.g. ["CONCH", "UNI2"])
        # Concatenate embeddings
        embs = [backbone_embeddings[m] for m in config]
        expert_emb = torch.cat(embs, dim=1)
        expert_embeddings_list.append(expert_emb)
        
        dim = sum(get_embedding_dim(m) for m in config)
        expert_dims.append(dim)
        
    # 5. Load Trained Model
    print("Loading trained model weights...")
    
    num_classes = len(UNIFIED_LABELS)
    is_ensemble = len(expert_configs) > 1

    ckpt_path = os.path.join(args.model_dir, "best_linear_probe.pt")
    print(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Loaded checkpoint is not a dictionary")
    if "model_state" not in checkpoint:
        raise RuntimeError("model_state not in checkpoint dictionary")

    if is_ensemble:
        hidden_dim = checkpoint.get("hidden_dim", HIDDEN_DIM)
        print(f"Ensemble: "
              f"hidden_dim={hidden_dim}")

        # Check for ablation by inspecting model keys
        state_keys = checkpoint["model_state"].keys()
        is_mlp_ablation = any(k.startswith("mlp.") for k in state_keys)
        is_gated_average_ablation = any(k.startswith("gate.") for k in state_keys)
        is_confidence_weighted_ablation = "confidence_temperature" in state_keys
        is_organ_ablation = "organ_embedding.weight" in state_keys
        
        organ_id_for_eval = None

        if is_mlp_ablation:
            print("Detected LogitMLPEnsemble (Ablation).")
            model = LogitMLPEnsemble(
                num_classes=num_classes,
                expert_embedding_dims=expert_dims,
                hidden_dim=hidden_dim
            ).to(device)
        elif is_gated_average_ablation:
            print("Detected GatedAverageEnsemble (Ablation).")
            model = GatedAverageEnsemble(
                num_classes=num_classes,
                expert_embedding_dims=expert_dims,
                hidden_dim=hidden_dim
            ).to(device)
        elif is_confidence_weighted_ablation:
            print("Detected ConfidenceWeightedEnsemble (Ablation).")
            model = ConfidenceWeightedEnsemble(
                num_classes=num_classes,
                expert_embedding_dims=expert_dims,
                hidden_dim=hidden_dim
            ).to(device)
        elif is_organ_ablation:
            print("Detected OrganCrossAttentionEnsemble (Ablation).")
            # Infer organ embedding parameters from checkpoint
            w = checkpoint["model_state"]["organ_embedding.weight"]
            num_organs_ckpt, organ_dim_ckpt = w.shape
            print(f"  Embedding shape: {w.shape} (num_organs={num_organs_ckpt}, dim={organ_dim_ckpt})")
            
            model = OrganCrossAttentionEnsemble(
                num_classes=num_classes,
                expert_embedding_dims=expert_dims,
                num_organs=num_organs_ckpt,
                organ_embedding_dim=organ_dim_ckpt,
                hidden_dim=hidden_dim
            ).to(device)
            
            try:
                organ_idx = ORGANS.index(organ_name)
                print(f"  Evaluation Organ: '{organ_name}' -> ID {organ_idx}")
                organ_id_for_eval = organ_idx
            except ValueError:
                print(f"  [WARN] Dataset '{organ_name}' not found in ORGANS list {ORGANS}. Defaulting to ID 0.")
                organ_id_for_eval = 0
        else:
            print("Detected CrossAttentionEnsemble.")
            model = CrossAttentionEnsemble(
                num_classes=num_classes,
                expert_embedding_dims=expert_dims,
                hidden_dim=hidden_dim
            ).to(device)
    else:
        model_type = checkpoint.get("model_type", "mlp")
        print(f"Single expert: model_type='{model_type}', emb_dim={expert_dims[0]}")
        if model_type == "logreg":
            model = nn.Linear(expert_dims[0], num_classes).to(device)
        else:
            model = LinearProbe(num_classes, expert_dims[0]).to(device)

    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    if missing:
        print(f"  Keys absent in checkpoint (fresh zero-init kept): {missing}")
    print("Model loaded.")
    model.eval()
    
    # 6. Run Inference
    print("Running inference...")
    
    # Create TensorDataset for efficient batching
    inference_ds = torch.utils.data.TensorDataset(*expert_embeddings_list)
    inference_loader = DataLoader(
        inference_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0 
    )
    
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Inference"):
            # Move inputs to device
            batch_expert_inputs = [x.to(device) for x in batch]
            
            if is_ensemble:
                if isinstance(model, OrganCrossAttentionEnsemble):
                    curr_batch_size = batch_expert_inputs[0].shape[0]
                    organs = torch.full((curr_batch_size,), organ_id_for_eval, dtype=torch.long, device=device)
                    logits = model(batch_expert_inputs, organ_ids=organs)
                else:
                    logits = model(batch_expert_inputs)
            else:
                # Single expert
                logits = model(batch_expert_inputs[0])
                
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.append(preds.cpu())
            
    all_preds = torch.cat(all_preds)
    
    # 7. Metrics
    acc = accuracy_score(test_labels, all_preds)
    print(f"\nAccuracy: {acc:.4f}")

    f1 = f1_score(test_labels, all_preds, average="weighted")
    print(f"F1 Score: {f1}")

    balanced_accuracy = balanced_accuracy_score(test_labels, all_preds)
    print(f"Balanced Accuracy: {balanced_accuracy}")
    
    # Save results
    print(f"Saving confusion matrix ...")
    cm = confusion_matrix(test_labels, all_preds, labels=list(range(num_classes)))
    
    save_dir = os.path.join(args.model_dir, "evaluation", args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=UNIFIED_LABELS)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(xticks_rotation=45, cmap="Blues", ax=ax, values_format='d')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    num_samples = len(all_preds)
        
    # Save raw predictions
    with open(os.path.join(save_dir, "predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "pred_label", "true_name", "pred_name"])
        for i in range(num_samples):
            writer.writerow([
                i, 
                test_labels[i].item(), 
                all_preds[i].item(),
                UNIFIED_LABELS[test_labels[i].item()],
                UNIFIED_LABELS[all_preds[i].item()]
            ])
            
    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    main()
