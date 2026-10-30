import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset


class SPIDERDataset(Dataset):
    """
    A PyTorch Dataset that loads stitched images on the fly from flat patch data.

    Assumes the following structure inside data_dir:
      - metadata.json
      - images/         (flat folder containing all patch images)

    The metadata.json file is expected to contain a list of records like:
      {
        "slide_id": "slide_identifier",
        "image_name": "central_patch_filename.png",
        "context_info": {
            "0_0": "patch_filename.png",
            "0_1": "patch_filename.png",
            ...
        },
        "class": "ClassName",
        "split": "train"  // or "test", etc.
      }

    For context sizes > 1, the dataset stitches patches together into one image (of size
    context_size * patch_size by context_size * patch_size). For context size 1, it simply loads
    the central patch.

    The dataset also builds a mapping from class names to numeric labels (and vice versa) which
    can be retrieved via get_class_to_label() and get_label_to_class().

    Args:
        data_dir (str): Directory containing metadata.json and an images folder.
        context_size (int): Context size to use: valid options are 1, 3, or 5.
        split (str): Which split to load ("train", "test", etc.). Only records with matching split are used.
        transform: Optional transform to apply to the stitched PIL image (e.g., a torchvision transform).
    """

    def __init__(self, data_dir, context_size=5, split="train", transform=None):
        self.data_dir = data_dir
        self.context_size = context_size
        self.split = split
        self.transform = transform

        # Fixed parameter: each patch is 224 pixels.
        self.patch_size = 224

        # The original flattened grid is assumed to be 5x5.
        self.full_grid_size = 5
        self.center_index = self.full_grid_size // 2  # For a 5x5 grid, this is 2.

        # Load metadata.
        metadata_file = os.path.join(data_dir, "metadata.json")
        with open(metadata_file, "r") as f:
            all_records = json.load(f)

        # Filter records by the split.
        self.records = [
            rec for rec in all_records if rec.get("split", "train") == self.split
        ]

        # Build class mappings.
        classes = sorted(set(rec["class"] for rec in self.records))
        self.class_to_label = {cls: idx for idx, cls in enumerate(classes)}
        self.label_to_class = {idx: cls for cls, idx in self.class_to_label.items()}

        # Define images directory.
        self.images_dir = os.path.join(data_dir, "images")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        # Compute bounds for the desired context.
        lower_bound = (self.full_grid_size - self.context_size) // 2
        upper_bound = lower_bound + self.context_size - 1

        # Create a new blank image for the stitched result.
        stitched_width = self.context_size * self.patch_size
        stitched_height = self.context_size * self.patch_size
        stitched_image = Image.new("RGB", (stitched_width, stitched_height))

        # Loop over grid coordinates to load and paste patches.
        for i in range(lower_bound, upper_bound + 1):
            for j in range(lower_bound, upper_bound + 1):
                # For the center patch, use record["image_name"]; otherwise, use context_info.
                if i == self.center_index and j == self.center_index:
                    patch_filename = record["image_name"]
                else:
                    key = f"{i}_{j}"
                    if key not in record["context_info"]:
                        # If a patch is missing, use a blank image.
                        patch = Image.new("RGB", (self.patch_size, self.patch_size))
                        paste_x = (j - lower_bound) * self.patch_size
                        paste_y = (i - lower_bound) * self.patch_size
                        stitched_image.paste(patch, (paste_x, paste_y))
                        continue
                    patch_filename = record["context_info"][key]

                patch_path = os.path.join(self.images_dir, patch_filename)
                try:
                    patch = Image.open(patch_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading patch {patch_path}: {e}")
                    patch = Image.new("RGB", (self.patch_size, self.patch_size))

                paste_x = (j - lower_bound) * self.patch_size
                paste_y = (i - lower_bound) * self.patch_size
                stitched_image.paste(patch, (paste_x, paste_y))

        # For context_size of 1, we simply use the center patch (without stitching)
        if self.context_size == 1:
            stitched_image = Image.open(
                os.path.join(self.images_dir, record["image_name"])
            ).convert("RGB")

        # Apply optional transform.
        if self.transform is not None:
            stitched_image = self.transform(stitched_image)

        # Get the numeric label.
        label = self.class_to_label[record["class"]]

        return stitched_image, label

    def get_class_to_label(self):
        return self.class_to_label

    def get_label_to_class(self):
        return self.label_to_class


if __name__ == "__main__":
    # Example usage:
    from torchvision import transforms

    # Define a simple transform that converts the PIL image to a tensor.
    transform = transforms.ToTensor()

    # Create dataset instance.
    dataset = SPIDERDataset(
        data_dir="/example/data_dir", context_size=5, split="train", transform=transform
    )

    print("Dataset size:", len(dataset))
    print("Class to label mapping:", dataset.get_class_to_label())
    print("Label to class mapping:", dataset.get_label_to_class())

    # Access one sample.
    img, label = dataset[0]
    print("Sample image shape:", img.shape, "Label:", label)