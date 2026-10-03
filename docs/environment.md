# Development environment

## Verified configuration

- Windows host with WSL2
- Ubuntu 22.04.5 LTS
- Python 3.10.12
- pip 26.2.1
- NVIDIA GeForce RTX 5080
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- PyTorch CUDA runtime: 12.8

Verified on 2026-10-03:
- CUDA matrix multiplication completed successfully.
- NumPy arrays were transferred to CUDA tensors successfully.
- Image-processing libraries imported successfully.
- pip check reported no broken requirements.

## Recreate the environment

These commands target Linux / WSL with a compatible NVIDIA driver.
Run them from the repository root using Python 3.10.

Create and activate a project environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pip==26.2.1
```

Install the CUDA-enabled PyTorch packages first:

```bash
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

Restore the recorded package versions:

```bash
python -m pip install -r requirements-frozen.txt
python -m pip check
```

The snapshot targets the configuration above.
It is not a universal dependency list for Windows, macOS or other GPUs.
The NVIDIA driver is managed outside the Python environment.

## Daily use

Activate the existing environment in each new terminal:

```bash
cd ~/projects/visual-quality-control
source .venv/bin/activate
```

Leave the environment:

```bash
deactivate
```

## Adding dependencies

Apply the PyTorch version constraints when installing packages:

```bash
python -m pip install -r requirements.txt -c constraints.txt
```

After changes, run relevant functional checks and pip check.
Then refresh requirements-frozen.txt and review its Git diff.

The .venv directory is excluded from Git.

## Install the project package

After installing the dependencies, run from the repository root:

```bash
python -m pip install --no-deps -e .
```

This makes the visual_quality package importable.
The editable installation uses the source files in this repository.

Check the package and data pipeline:

```bash
python -c "import visual_quality; print(visual_quality.__file__)"
python scripts/check_dataloader.py
```

The DataLoader check requires the downloaded bottle dataset
and the committed split manifest.

When refreshing the dependency snapshot, exclude editable packages:

```bash
python -m pip freeze --exclude-editable > requirements-frozen.txt
```

The local project is installed separately with the editable command above.
