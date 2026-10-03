from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Find the project directory based on this script's location.
project_root = Path(__file__).resolve().parents[1]
dataset_root = project_root / "data" / "mvtec_ad" / "bottle"

# Match an image with its ground-truth mask.
image_path = dataset_root / "test" / "broken_large" / "000.png"
mask_path = dataset_root / "ground_truth" / "broken_large" / "000_mask.png"

# Load the image as RGB and the mask as a single-channel image.
with Image.open(image_path) as file:
    image = np.array(file.convert("RGB"))

with Image.open(mask_path) as file:
    mask = np.array(file.convert("L"))

# Check that the image and mask can be compared pixel by pixel.
if image.shape[:2] != mask.shape:
    raise ValueError("Image and mask dimensions do not match.")

mask_values = np.unique(mask)

if not np.isin(mask_values, [0, 255]).all():
    raise ValueError(f"Unexpected mask values: {mask_values}")

# True marks pixels belonging to a defect.
defect = mask > 0

# Blend the defective region with red, keeping other pixels unchanged.
overlay = image.copy()
red = np.array([255, 0, 0], dtype=np.float32)
overlay[defect] = (
    0.6 * image[defect] + 0.4 * red
).astype(np.uint8)

# Create a three-panel preview.
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image)
axes[0].set_title("Original image")

axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Ground-truth mask")

axes[2].imshow(overlay)
axes[2].set_title("Annotated defect")

for axis in axes:
    axis.axis("off")

fig.tight_layout()

output_dir = project_root / "artifacts" / "previews"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "bottle_broken_large_000.png"

fig.savefig(output_path, dpi=150)
plt.close(fig)

print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Mask shape:", mask.shape)
print("Mask values:", mask_values.tolist())
print("Defect pixels:", int(defect.sum()))
print("Saved preview:", output_path)
