# Reproduce the project

## Status and scope

This recipe targets the recorded Linux/WSL environment: Python 3.10,
RTX 5080, PyTorch 2.11.0+cu128 and TensorRT 10.13.3.9.post1.
The existing demo, numerical comparisons, API restart and installed-wheel GUI
checks passed on that machine. The revised workflow passed GPU verification on 2026-10-03, starting
from the existing checkpoint: new ONNX export, pipeline validation,
TensorRT build, optimized-pipeline parity, localization calibration
and two real API lifecycles. Evidence: verification/reproduction/README.md.
A full fresh-environment replay including memory-bank construction
has not yet been recorded.
Do not describe the revised recipe as independently reproduced until that run
has completed and its reports have been archived.

Commands below run in a WSL terminal, from the repository root. Stop at the
first failed command. They are a new-experiment recipe, not commands to replace
the existing demo's artifacts. FP16 and INT8 are outside this release's scope.

## 1. Clone and install (new checkout only)

Prerequisites: Git, Python 3.10 with venv support, working WSL GPU access and a
compatible Windows NVIDIA driver. The driver is outside `.venv`. Dataset and
pretrained-backbone downloads require network access.

```bash
mkdir -p ~/projects
git clone https://github.com/JJan010/visual-quality-control.git
cd visual-quality-control
git rev-parse HEAD
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install pip==26.2.1
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-frozen.txt -c constraints.txt
python -m pip install --no-deps -e .
python -m pip check
```

Use a commit containing the revised scripts in this recipe. The frozen snapshot
includes API and TensorRT dependencies. Installing only `requirements.txt` does
not include the API dependency list. GPU Python libraries are installed in this
project's venv; model/data caches can still live outside it.

`requirements-frozen.txt` is a snapshot of installed package versions, not a
hash-locked, platform-independent dependency specification. Availability of the
recorded distributions and build tools is still required. Do not replace missing
versions silently: document a new environment and rerun the relevant checks.
The project checks selected library versions when loading a checkpoint.

Check GPU execution:

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
a = torch.randn(64, 64, device='cuda')
b = a @ a.T
torch.cuda.synchronize()
assert torch.isfinite(b).all().item()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print('GPU: OK')
PY
```

## 2. Obtain the dataset

In a browser, open the official [MVTec AD dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
and download the original MVTec AD **bottle** category archive. Follow its license
terms; see `docs/dataset.md`. The dataset is not supplied by this repository.

For the original author's Windows account, after downloading `bottle.tar.xz`:

```bash
mkdir -p data/downloads data/mvtec_ad
cp /mnt/c/Users/janec/Downloads/bottle.tar.xz data/downloads/
tar -tJf data/downloads/bottle.tar.xz | head -n 25
tar -xJf data/downloads/bottle.tar.xz -C data/mvtec_ad
python scripts/create_split.py
```

Other users must substitute their actual download path. Extract into a fresh
dataset directory. Expected layout: `data/mvtec_ad/bottle/train/good`, `test`,
`ground_truth`. Counts: 209 normal training images, 63 defective and 20 normal
test images. The committed split reserves 167 normal images for the memory bank
and 42 normal images for calibration. `create_split.py` rejects a conflicting
existing split instead of replacing it.

## 3. Build and evaluate a new baseline

Set these variables once in the same WSL terminal. All output directories must
be new. If repeating the experiment, change `reproduction_v1` to another name.
Do not pick the newest directory automatically or reuse the historical hashes.

```bash
export VQC_RUN="$PWD/artifacts/runs/reproduction_v1"
export VQC_EXPORT="$VQC_RUN/exports/features_onnx_v1"
export VQC_ENGINE="$VQC_EXPORT/tensorrt_fp32_v1"
export VQC_RUNTIME_CONFIG="$PWD/configs/runtime/patchcore_reproduced.json"
export VQC_LOC_DIR="$VQC_RUN/localization_calibration/reproduction_v1"

python scripts/build_patchcore_bank.py --output-dir "$VQC_RUN"
python scripts/calibrate_patchcore.py --run-dir "$VQC_RUN"
python scripts/evaluate_patchcore.py --run-dir "$VQC_RUN"
```

The bank builder uses a pretrained backbone and a coreset from the training
split. Classification calibration uses only the normal validation split.
Evaluation writes `evaluation/metrics.json` and `test_scores.csv` for this run;
these also provide the recorded decision reference for deployment checks.

The baseline scripts preserve their original precision policy. Deployment
comparisons explicitly use IEEE FP32. Their decision comparison against the
saved baseline must pass; a failure needs investigation, not an edited report.
The fixed seed does not promise identical checkpoints, coreset selection,
thresholds, latency or scores across hardware/library changes.

## 4. Export ONNX and validate its complete pipeline

```bash
python scripts/export_patchcore_features.py \
  --run-dir "$VQC_RUN" --output-dir "$VQC_EXPORT"

python scripts/check_patchcore_onnx_iobinding.py \
  --run-dir "$VQC_RUN" --export-dir "$VQC_EXPORT"
```

The first command creates `feature_extractor.onnx` and `export_check.json` with
status `exported_pending_pipeline_validation`. It validates the graph, feature
shapes, finite values and recorded GPU convolution execution. CPU PyTorch versus
CUDA ORT feature differences retain the original `atol=rtol=1e-4` diagnostic;
failure of this diagnostic is recorded and is not mislabeled as a pass.

The second command is the acceptance gate. It checks I/O binding at batch sizes
1, 3 and 8, then compares image scores and anomaly maps of the complete pipeline
against live IEEE FP32 PyTorch on all 83 test images at batch sizes 1 and 8.
It retains `atol=rtol=1e-4`, requires finite values, unchanged decisions and
agreement with the recorded baseline decisions. The final incomplete batch is
also included. This gate validates pipeline behavior, not bitwise feature parity.

Only after success are these files published alongside this specific ONNX:

- `pipeline_validation.json`: numerical checks, precision, model identity and
  hashes of supporting files;
- `runtime_manifest.json`: model configuration and hashes linking it to ONNX and
  the validation report.

Do not manually change statuses or hashes. The test split is also used for
engineering parity checks; it is not an untouched production holdout. Neither
classification nor localization thresholds are chosen from its labels.

## 5. Build TensorRT and verify it

```bash
python scripts/build_patchcore_tensorrt.py \
  --run-dir "$VQC_RUN" --export-dir "$VQC_EXPORT" \
  --output-dir "$VQC_ENGINE" --opt-batch 1

python scripts/check_patchcore_tensorrt.py \
  --run-dir "$VQC_RUN" --export-dir "$VQC_EXPORT" \
  --engine-dir "$VQC_ENGINE"

python scripts/check_patchcore_optimized.py \
  --run-dir "$VQC_RUN" --engine-dir "$VQC_ENGINE"
```

The builder verifies the current export's approval and its evidence hashes.
It does not use the historical report under `docs/experiments`. A built engine
is not automatically a numerically accepted engine: both subsequent checks
must pass before configuring the demo. The second check covers the deployed
TensorRT + separable-blur combination. These tests do not certify performance.

The engine uses FP32 with TF32 disabled and batch range 1–8, optimum 1. Build it
on the target GPU. The project's loader checks GPU name and TensorRT version;
do not assume a `.plan` is portable to a different machine or TensorRT release.
Historical benchmark values must not be attributed to the new engine without
measuring it separately.

## 6. Create a separate runtime configuration

Keep the existing `patchcore_tensorrt.json` and its demo artifacts. Generate a new
configuration using the explicit paths above:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ['VQC_RUNTIME_CONFIG']).resolve()
run = Path(os.environ['VQC_RUN']).resolve()
engine = Path(os.environ['VQC_ENGINE']).resolve()
if not (run / 'calibration/threshold.json').is_file():
    raise SystemExit('Missing image calibration')
if not (engine / 'build_report.json').is_file():
    raise SystemExit('Missing engine build report')
config = {
    'schema_version': 1,
    'backend': 'tensorrt',
    'device': 'cuda:0',
    'precision': 'ieee',
    'blur_backend': 'separable_1d',
    'run_dir': os.path.relpath(run, path.parent),
    'engine_dir': os.path.relpath(engine, path.parent),
}
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('x', encoding='utf-8') as file:
    file.write(json.dumps(config, indent=2) + '\n')
print('Runtime config:', path)
PY
```

Relative paths are resolved from the configuration's directory, not the shell's
working directory. The `x` mode intentionally refuses to overwrite a config.

## 7. Calibrate localization for this exact runtime

```bash
python scripts/calibrate_localization.py \
  --config "$VQC_RUNTIME_CONFIG" --output-dir "$VQC_LOC_DIR"
```

This uses the maxima of anomaly maps on 42 normal validation images and the
95th percentile (`higher`). It does not change the classification threshold.
The output is bound to the runtime config bytes, checkpoint, split and runtime
metadata. Finalize paths/config first, then calibrate. Moving the installation
or editing/reformatting the config may invalidate calibration; regenerate it
for the final deployment instead of editing hashes.

Contours are thresholded model estimates, not exact defect boundaries. The
historical experiment had high pixel recall and broader predicted regions;
that limitation remains relevant to the GUI's wording.

## 8. Run CLI and GUI

```bash
python scripts/predict_image.py \
  --config "$VQC_RUNTIME_CONFIG" \
  --image data/mvtec_ad/bottle/test/broken_large/011.png --save-map

VQC_CONFIG="$VQC_RUNTIME_CONFIG" \
VQC_LOCALIZATION_CALIBRATION="$VQC_LOC_DIR/threshold.json" \
python -m uvicorn visual_quality.api.app:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

Open **http://127.0.0.1:8000/** in the Windows browser. Upload a bottle image,
run inspection and view the decision, score, timings and optional contour.
API documentation: **http://127.0.0.1:8000/docs**. Stop the server with Ctrl+C.
Use one worker and no auto-reload for this application. Keep it on localhost.

In a new terminal, reactivate `.venv` and set the selected config/calibration
paths again. Existing diagnostic scripts with defaults tied to the original
experiment are historical helpers; do not silently use their defaults for this
new run. In particular, `check_api_restart.py` accepts `--config` and
`--calibration-file`, but its two fixed expected decisions are also an explicit
regression expectation, not a guarantee for an arbitrary retrained model.

## 9. Evidence and packaging

Archive small reports under `docs/verification` and record the commit, hardware,
versions and chosen paths. Keep `.venv`, images, model weights, ONNX files and
TensorRT engines out of ordinary Git commits. Preserve originals of historical
reports. A new run's results should be recorded as new results.

```bash
python -m unittest discover -s tests -p 'test_export_validation.py' -v
python -m unittest discover -s tests -p 'test_api_worker.py' -v
python -m pip wheel . --no-deps --wheel-dir artifacts/packaging
python scripts/check_installed_wheel.py
```

The wheel includes Python modules and GUI assets. It does not include dataset,
trained models or runtime configs, and dependency installation is handled by the
requirements procedure above. The wheel check uses existing venv dependencies;
it is not a clean-environment installation test. Build tools are not pinned by
the Python 3.10 `pip freeze` snapshot.

## References

- [MVTec AD and its original paper](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [pip freeze scope](https://pip.pypa.io/en/stable/cli/pip_freeze/)
- [TensorRT support and portability](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- [Setuptools package data](https://setuptools.pypa.io/en/latest/userguide/datafiles.html)
