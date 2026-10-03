# Reproduction verification

Verified on 2026-10-03 in the existing WSL/Python environment on RTX 5080.

## Verified scope

Starting from the existing PatchCore checkpoint:

- Created a new ONNX export.
- Generated per-export pipeline approval without the historical approval file.
- Passed I/O binding checks at batch sizes 1, 3 and 8.
- Passed score and map parity on all 83 test images at batch sizes 1 and 8.
- Built and deserialized a new TensorRT FP32 engine with TF32 disabled.
- Passed TensorRT parity and the combined TensorRT + separable-blur checks.
- Observed zero changed image decisions in the parity checks.
- Calibrated localization using 42 normal validation images.
- Started, queried and gracefully stopped the real API twice.
- Passed prediction agreement after restart.

Localization threshold, rounded: 31.18199730.
Calibration images with any flagged pixel: 2/42.
Image classification threshold: 32.44375991821289.

## Evidence and limits

The adjacent reports preserve the artifact identities and numerical results.
runtime_config.json is an archived copy; its relative paths are interpreted
from its original location under configs/runtime, not from this directory.

The CPU-versus-CUDA feature comparison remains a diagnostic and did not pass
its original tolerance. Complete-pipeline acceptance was checked separately
at the unchanged score/map tolerances.

This verifies artifact rebuilding from an existing checkpoint in the existing
environment. It does not verify a clean dependency installation or rebuilding
the memory bank from scratch. No new performance benchmark was performed.
Historical latency measurements must not be attributed to this new engine.
