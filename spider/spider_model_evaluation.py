import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Add parent directory to path to import modules from Model Training root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import (
    SPIDER_SKIN_DATA_ROOT, SPIDER_THORAX_DATA_ROOT, 
    SPIDER_BREAST_DATA_ROOT, SPIDER_COLORECTAL_DATA_ROOT,
    SPIDER_SKIN_TO_UNIFIED, SPIDER_THORAX_TO_UNIFIED, 
    SPIDER_BREAST_TO_UNIFIED, SPIDER_COLORECTAL_TO_UNIFIED,
    UNIFIED_NAME_TO_ID, UNIFIED_LABELS
)
from spider.spider_dataset import SPIDERDataset

# Dataset specific configs
DATASET_CONFIGS = {
    "skin": {
        "root": SPIDER_SKIN_DATA_ROOT,
        "mapping": SPIDER_SKIN_TO_UNIFIED,
        "hf_model": "histai/SPIDER-skin-model"
    },
    "thorax": {
        "root": SPIDER_THORAX_DATA_ROOT,
        "mapping": SPIDER_THORAX_TO_UNIFIED,
        "hf_model": "histai/SPIDER-thorax-model"
    },
    "breast": {
        "root": SPIDER_BREAST_DATA_ROOT,
        "mapping": SPIDER_BREAST_TO_UNIFIED,
        "hf_model": "histai/SPIDER-breast-model"
    },
    "colorectal": {
        "root": SPIDER_COLORECTAL_DATA_ROOT,
        "mapping": SPIDER_COLORECTAL_TO_UNIFIED,
        "hf_model": "histai/SPIDER-colorectal-model"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Evaluate Open Source SPIDER Models")
    parser.add_argument("--dataset", type=str, required=True, choices=["skin", "thorax", "breast", "colorectal"],
                        help="Dataset to evaluate (controls data root and label mapping)")
    parser.add_argument("--model", type=str, default=None, choices=["skin", "thorax", "breast", "colorectal"],
                        help="Which SPIDER model to use (defaults to matching --dataset). "
                             "Set this to evaluate a model on a different dataset, e.g. --model colorectal --dataset skin.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    args = parser.parse_args()

    dataset_key = args.dataset.lower()
    model_key   = args.model.lower() if args.model else dataset_key

    config       = DATASET_CONFIGS[dataset_key]          # data root + label mapping
    model_config = DATASET_CONFIGS[model_key]            # hf_model to load
    device = torch.device(args.device)

    # Set deterministic seeds
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    np.random.seed(100)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run_label = f"{args.dataset}_with_{model_key}_model" if model_key != dataset_key else args.dataset
    print(f"Evaluating dataset='{args.dataset}'  model='{model_config['hf_model']}'")
    print(f"Data Root: {config['root']}")

    # 1. Load Model and Processor
    try:
        print(f"Loading model: {model_config['hf_model']} ...")
        model = AutoModel.from_pretrained(model_config['hf_model'], trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_config['hf_model'], trust_remote_code=True)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model {model_config['hf_model']}: {e}")
        return

    # 2. Setup Dataset
    def hf_transform(image):
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)

    # Note: Context size 5 is required for 1120x1120 input as per model card
    dataset = SPIDERDataset(
        data_dir=config['root'],
        context_size=1,
        split="test",
        transform=hf_transform
    )
    model_dataset = SPIDERDataset(data_dir=model_config["root"], context_size=1, split="test", transform=hf_transform)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Test dataset size: {len(dataset)}")
    
    # 3. Inference Loop
    y_true_unified = []
    y_pred_unified = []
    
    # Dataset ID -> Dataset Label Name (for ground-truth labels)
    dataset_id2label = dataset.get_label_to_class()
    model_id2label = model_dataset.get_label_to_class()

    # Mapping: Dataset Label Name -> Unified Label Name (for ground-truth labels)
    ds_mapping = config['mapping']
    # Mapping: Model Label Name -> Unified Label Name (for predictions, may differ when model != dataset)
    model_mapping = model_config['mapping']

    print("Starting inference...")
    with torch.no_grad():
        for batch_imgs, batch_lbls in tqdm(dataloader):
            batch_imgs = batch_imgs.to(device)

            # Forward pass
            outputs = model(pixel_values=batch_imgs)
            
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            batch_lbls = batch_lbls.numpy()
            
            for i in range(len(preds)):
                # Get prediction string from Model's own label space
                pred_idx = preds[i]
                pred_label_name = model_id2label[pred_idx]

                # Get ground truth string from Dataset Metadata
                gt_idx = batch_lbls[i]
                gt_label_name = dataset_id2label[gt_idx]

                # Map both to Unified Taxonomy to compare apples-to-apples
                try:
                    unified_pred_name = model_mapping.get(pred_label_name)
                    unified_gt_name = ds_mapping.get(gt_label_name)
                    
                    if unified_pred_name and unified_gt_name:
                        y_pred_unified.append(UNIFIED_NAME_TO_ID[unified_pred_name])
                        y_true_unified.append(UNIFIED_NAME_TO_ID[unified_gt_name])
                    else:
                        print(f"Unknown label or prediction. Predicted: {pred_label_name}, Label: {gt_label_name}")
                        if unified_gt_name:
                            y_true_unified.append(UNIFIED_NAME_TO_ID[unified_gt_name])
                            y_pred_unified.append(-1)

                except Exception:
                    continue

    # 4. Evaluation Metrics
    accuracy = accuracy_score(y_true_unified, y_pred_unified)
    print(f"Unified Accuracy: {accuracy:.4f}")

    f1 = f1_score(y_true_unified, y_pred_unified, average="weighted")
    print(f"F1 Score: {f1}")

    balanced_accuracy = balanced_accuracy_score(y_true_unified, y_pred_unified)
    print(f"Balanced Accuracy: {balanced_accuracy}")

    report = classification_report(
        y_true_unified, 
        y_pred_unified, 
        target_names=UNIFIED_LABELS, 
        labels=range(len(UNIFIED_LABELS)),
        zero_division=0
    )
    print("Classification Report:")
    print(report)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"{run_label}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_config['hf_model']}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Unified Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    
    # Save predictions and ground truth for downstream analysis
    results_df = pd.DataFrame({
        'y_true': y_true_unified,
        'y_pred': y_pred_unified
    })
    results_csv_path = os.path.join(args.output_dir, f"{run_label}_predictions.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Predictions saved to {results_csv_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true_unified, y_pred_unified, labels=range(len(UNIFIED_LABELS)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=UNIFIED_LABELS)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title(f"Confusion Matrix: dataset={args.dataset}  model={model_key} (Unified)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{run_label}_cm.png"))
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
