"""
Evaluate SPIDER models on the LubLung dataset.

Usage:
    python Spider/spider_model_evaluation_lung.py --model thorax --output_dir outputs/SPIDER/lung_eval

This script:
1. Loads a selected SPIDER model (skin, thorax, breast, or colorectal)
2. Runs inference on all LubLung tiles
3. Maps both SPIDER predictions and LubLung ground truth to unified taxonomy
4. Reports classification metrics and saves results
"""

import sys
import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, 
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path to import modules from Model Training root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import (
    SPIDER_SKIN_DATA_ROOT, SPIDER_THORAX_DATA_ROOT, 
    SPIDER_BREAST_DATA_ROOT, SPIDER_COLORECTAL_DATA_ROOT,
    UNIFIED_NAME_TO_ID, UNIFIED_LABELS)

from dataset_mappings import (
    LUB_LUNG_TO_UNIFIED,
    SPIDER_SKIN_TO_UNIFIED,
    SPIDER_THORAX_TO_UNIFIED,
    SPIDER_BREAST_TO_UNIFIED,
    SPIDER_COLORECTAL_TO_UNIFIED,
)

# LubLung dataset path
LUBLUNG_DATA_ROOT = os.path.join("Data", "LubLung", "LubLung")

# SPIDER model configurations
SPIDER_MODELS = {
    "skin": {
        "data_root": SPIDER_SKIN_DATA_ROOT,
        "mapping": SPIDER_SKIN_TO_UNIFIED,
        "hf_model": "histai/SPIDER-skin-model"
    },
    "thorax": {
        "data_root": SPIDER_THORAX_DATA_ROOT,
        "mapping": SPIDER_THORAX_TO_UNIFIED,
        "hf_model": "histai/SPIDER-thorax-model"
    },
    "breast": {
        "data_root": SPIDER_BREAST_DATA_ROOT,
        "mapping": SPIDER_BREAST_TO_UNIFIED,
        "hf_model": "histai/SPIDER-breast-model"
    },
    "colorectal": {
        "data_root": SPIDER_COLORECTAL_DATA_ROOT,
        "mapping": SPIDER_COLORECTAL_TO_UNIFIED,
        "hf_model": "histai/SPIDER-colorectal-model"
    }
}

# LubLung classes to include (excluding 03_MIXED as per dataset_mappings.py)
LUBLUNG_CLASSES = [
    "01_TUMOR",
    "02_STROMA",
    "04_IMMUNE",
    "05_VESSEL",
    "06_BRONCHI",
    "07_NECROSIS",
    "08_LUNG",
    "09_BACKGROUND",
]


class LubLungDataset(Dataset):
    """Simple folder-based dataset for LubLung tiles."""
    
    def __init__(self, data_root: str, transform=None):
        """
        Args:
            data_root: Path to LubLung folder containing class subfolders
            transform: Optional transform to apply to images
        """
        self.data_root = data_root
        self.transform = transform
        self.samples = []  # List of (image_path, class_name)
        
        # Collect all samples from included classes
        for class_name in LUBLUNG_CLASSES:
            class_path = os.path.join(data_root, class_name)
            if not os.path.isdir(class_path):
                print(f"Warning: Class folder not found: {class_path}")
                continue
                
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.samples.append((
                        os.path.join(class_path, fname),
                        class_name
                    ))
        
        print(f"Loaded {len(self.samples)} samples from LubLung dataset")
        
        # Print class distribution
        class_counts = {}
        for _, class_name in self.samples:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        print("Class distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Apply transform
        if self.transform is not None:
            img = self.transform(img)
        
        return img, class_name


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SPIDER models on the LubLung dataset"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=["skin", "thorax", "breast", "colorectal"],
        help="SPIDER model to use for evaluation"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save results"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=LUBLUNG_DATA_ROOT,
        help="Path to LubLung dataset folder"
    )
    args = parser.parse_args()

    config = SPIDER_MODELS[args.model]
    device = torch.device(args.device)

    # Set deterministic seeds
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    np.random.seed(100)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"{'='*60}")
    print(f"Evaluating SPIDER-{args.model} model on LubLung dataset")
    print(f"Model: {config['hf_model']}")
    print(f"Data Root: {args.data_root}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # 1. Load Model and Processor
    try:
        print(f"\nLoading model: {config['hf_model']} ...")
        model = AutoModel.from_pretrained(config['hf_model'], trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(config['hf_model'], trust_remote_code=True)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model {config['hf_model']}: {e}")
        return


    metadata_file = os.path.join(config["data_root"], "metadata.json")
    
    with open(metadata_file, "r") as f:
        records = json.load(f)

    # Build class mappings.
    classes = sorted(set(rec["class"] for rec in records))
    spider_id2label = {idx: cls for idx, cls in enumerate(classes)}

    print(f"\nSPIDER model labels: {spider_id2label}")

    # 2. Setup Dataset with SPIDER preprocessing
    def hf_transform(image):
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)

    dataset = LubLungDataset(
        data_root=args.data_root,
        transform=hf_transform
    )

    if len(dataset) == 0:
        print("Error: No samples found in dataset!")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # 3. Inference Loop
    y_true_unified = []
    y_pred_unified = []
    y_true_lublung = []  # Original LubLung labels
    y_pred_spider = []   # Original SPIDER predictions
    all_logits = []      # Raw logits from model
    all_probs = []       # Softmax probabilities
    all_confidence = []  # Top-1 confidence scores
    
    spider_mapping = config['mapping']
    
    # Track unmapped predictions
    unmapped_predictions = {}

    print("\nStarting inference...")
    with torch.no_grad():
        for batch_imgs, batch_classes in tqdm(dataloader, desc="Processing"):
            batch_imgs = batch_imgs.to(device)

            # Forward pass
            outputs = model(pixel_values=batch_imgs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Get logits and probs as numpy
            batch_logits = logits.cpu().numpy()
            batch_probs = probs.cpu().numpy()

            for i in range(len(preds)):
                # Get SPIDER prediction
                pred_idx = preds[i]
                spider_pred_name = spider_id2label[pred_idx]
                
                # Get LubLung ground truth
                lublung_class = batch_classes[i]
                
                # Store original labels
                y_true_lublung.append(lublung_class)
                y_pred_spider.append(spider_pred_name)
                
                # Store logits and probs for this sample
                all_logits.append(batch_logits[i])
                all_probs.append(batch_probs[i])
                all_confidence.append(float(batch_probs[i].max()))
                
                # Map LubLung ground truth to unified
                unified_gt_name = LUB_LUNG_TO_UNIFIED.get(lublung_class)
                if unified_gt_name is None:
                    print(f"Warning: LubLung class '{lublung_class}' not in mapping")
                    continue
                
                # Map SPIDER prediction to unified
                unified_pred_name = spider_mapping.get(spider_pred_name)
                if unified_pred_name is None:
                    # Track unmapped predictions
                    unmapped_predictions[spider_pred_name] = unmapped_predictions.get(spider_pred_name, 0) + 1
                    # Use a placeholder for unmapped predictions
                    unified_pred_name = "Background"  # Default fallback
                
                y_true_unified.append(UNIFIED_NAME_TO_ID[unified_gt_name])
                y_pred_unified.append(UNIFIED_NAME_TO_ID[unified_pred_name])

    print(f"\nProcessed {len(y_true_unified)} samples")
    
    if unmapped_predictions:
        print(f"\nWarning: Unmapped SPIDER predictions (mapped to 'Background'):")
        for pred, count in sorted(unmapped_predictions.items(), key=lambda x: -x[1]):
            print(f"  {pred}: {count}")

    # 4. Evaluation Metrics
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS (Unified Taxonomy)")
    print(f"{'='*60}")
    
    accuracy = accuracy_score(y_true_unified, y_pred_unified)
    print(f"Accuracy: {accuracy:.4f}")

    f1 = f1_score(y_true_unified, y_pred_unified, average="weighted", zero_division=0)
    print(f"F1 Score (weighted): {f1:.4f}")

    f1_macro = f1_score(y_true_unified, y_pred_unified, average="macro", zero_division=0)
    print(f"F1 Score (macro): {f1_macro:.4f}")

    balanced_acc = balanced_accuracy_score(y_true_unified, y_pred_unified)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    # Classification report
    report = classification_report(
        y_true_unified, 
        y_pred_unified, 
        target_names=UNIFIED_LABELS, 
        labels=range(len(UNIFIED_LABELS)),
        zero_division=0
    )
    print("\nClassification Report:")
    print(report)

    # 5. Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save text report
    report_path = os.path.join(args.output_dir, f"lublung_{args.model}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"SPIDER Model Evaluation on LubLung Dataset\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: {config['hf_model']}\n")
        f.write(f"Dataset: LubLung\n")
        f.write(f"Total Samples: {len(y_true_unified)}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score (weighted): {f1:.4f}\n")
        f.write(f"F1 Score (macro): {f1_macro:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
        if unmapped_predictions:
            f.write(f"\n\nUnmapped SPIDER predictions (mapped to 'Background'):\n")
            for pred, count in sorted(unmapped_predictions.items(), key=lambda x: -x[1]):
                f.write(f"  {pred}: {count}\n")
    
    print(f"\nReport saved to {report_path}")

    # Save predictions CSV with logits and probabilities
    results_data = {
        'lublung_class': y_true_lublung,
        'spider_prediction': y_pred_spider,
        'y_true_unified': y_true_unified,
        'y_pred_unified': y_pred_unified,
        'y_true_name': [UNIFIED_LABELS[y] for y in y_true_unified],
        'y_pred_name': [UNIFIED_LABELS[y] for y in y_pred_unified],
        'confidence': all_confidence,
    }
    
    # Add logit columns for each SPIDER class
    num_spider_classes = len(spider_id2label)
    for class_idx in range(num_spider_classes):
        class_name = spider_id2label[class_idx]
        results_data[f'logit_{class_name}'] = [logits[class_idx] for logits in all_logits]
        results_data[f'prob_{class_name}'] = [probs[class_idx] for probs in all_probs]
    
    results_df = pd.DataFrame(results_data)
    results_csv_path = os.path.join(args.output_dir, f"lublung_{args.model}_predictions.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Predictions saved to {results_csv_path}")

    # 6. Confusion Matrix
    cm = confusion_matrix(y_true_unified, y_pred_unified, labels=range(len(UNIFIED_LABELS)))
    
    # Normalized confusion matrix
    cm_normalized = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    # Raw counts
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=UNIFIED_LABELS)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, values_format='d')
    plt.title(f"Confusion Matrix: SPIDER-{args.model} on LubLung (Counts)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"lublung_{args.model}_cm.png"), dpi=300)
    plt.close()
    
    # Normalized
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=UNIFIED_LABELS)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, values_format='.2f')
    plt.title(f"Confusion Matrix: SPIDER-{args.model} on LubLung (Normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"lublung_{args.model}_cm_normalized.png"), dpi=300)
    plt.close()

    print(f"Confusion matrices saved to {args.output_dir}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
