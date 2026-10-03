import json
import random
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
dataset_root = project_root / "data" / "mvtec_ad" / "bottle"

seed = 42
validation_count = 42

# Start from a stable, sorted list of normal training images.
files = sorted((dataset_root / "train" / "good").glob("*.png"))

if len(files) != 209:
    raise ValueError(f"Expected 209 training images, found {len(files)}.")

# Store paths relative to the bottle directory.
relative_paths = [
    file.relative_to(dataset_root).as_posix()
    for file in files
]

# Shuffle using a local random-number generator.
rng = random.Random(seed)
rng.shuffle(relative_paths)

validation_paths = sorted(relative_paths[:validation_count])
train_paths = sorted(relative_paths[validation_count:])

# Verify that the two lists are disjoint and cover all input paths.
train_set = set(train_paths)
validation_set = set(validation_paths)

if train_set & validation_set:
    raise ValueError("Training and validation paths overlap.")

if train_set | validation_set != set(relative_paths):
    raise ValueError("The split does not cover all input paths.")

manifest = {
    "schema_version": 1,
    "dataset": "MVTec AD",
    "category": "bottle",
    "dataset_root": "data/mvtec_ad/bottle",
    "seed": seed,
    "train": train_paths,
    "validation": validation_paths,
}

output_path = project_root / "configs" / "splits" / "bottle_v1.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Prevent silently replacing a previously recorded split.
if output_path.exists():
    existing = json.loads(output_path.read_text(encoding="utf-8"))
    if existing != manifest:
        raise ValueError("A different split already exists at this path.")
    print("Existing split matches. No changes made.")
else:
    output_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Split saved.")

print("Training images:", len(train_paths))
print("Validation images:", len(validation_paths))
print("Overlapping paths:", len(train_set & validation_set))
print("Manifest:", output_path)
