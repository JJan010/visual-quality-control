# PatchCore: PyTorch, ONNX Runtime and TensorRT FP32

## Scope and correctness

The experiment replaces only the feature extractor.
Normalization, memory-bank search and anomaly-map generation remain
in PyTorch. Both anomaly-map filtering and the decision threshold
remain unchanged.

Configuration:

- NVIDIA GeForce RTX 5080.
- PyTorch IEEE FP32.
- ONNX Runtime CUDA with use_tf32=0 and GPU I/O Binding.
- TensorRT 10.13.3.9.post1, FP32 with TF32 disabled.
- Original 2D Gaussian blur.
- TensorRT batch profile: min=1, opt=1, max=8.
- TensorRT adapter uses a dedicated CUDA stream and synchronizes
  before returning GPU outputs.

TensorRT pipeline outputs were compared against both PyTorch and
ONNX Runtime on all 83 test images, separately at batch sizes 1 and 8.
Scores and maps passed atol=1e-4, rtol=1e-4.
No decisions changed between implementations or relative to the
saved baseline.

See tensorrt_parity.json for exact errors and artifact hashes.

## Measurement method

Each backend received 30 warmup calls and 120 measured calls per
batch size. All three backends processed identical inputs within
each measurement round.

Each of the six backend orderings occurred 20 times. Their sequence
was shuffled using seed 42.

The timed scope starts with a prepared CPU RGB batch and includes
transfer to GPU, normalization, full PatchCore inference, and transfer
of scores, labels and anomaly maps back to CPU.

Disk reads, decoding, resizing, CPU batch creation, model loading
and backend selection were excluded. CUDA was synchronized at the
timing boundaries.

## Results

| Batch | Backend | Mean ms | p50 ms | p95 ms | Serial images/s |
|---|---|---:|---:|---:|---:|
| 1 | PyTorch | 7.370 | 6.937 | 8.668 | 135.69 |
| 1 | ONNX Runtime | 7.173 | 6.814 | 8.189 | 139.40 |
| 1 | TensorRT FP32 | 4.148 | 3.848 | 4.975 | 241.09 |
| 8 | PyTorch | 25.754 | 25.855 | 26.579 | 310.63 |
| 8 | ONNX Runtime | 25.799 | 25.923 | 26.741 | 310.09 |
| 8 | TensorRT FP32 | 20.673 | 20.795 | 21.547 | 386.98 |

Relative to PyTorch in this run:

- Batch 1: TensorRT achieved 1.777x speedup, reducing mean latency
  by approximately 43.7%.
- Batch 8: TensorRT achieved 1.246x speedup, reducing mean latency
  by approximately 19.7%.

ONNX Runtime performed close to PyTorch.

## Interpretation and limits

TensorRT FP32 improved complete-pipeline latency while preserving
the tested numerical and decision criteria.

These results describe one short run on one machine. They are not
a service throughput guarantee or an isolated feature-extractor
benchmark. All backend instances were resident during measurement.

The TensorRT profile was optimized for batch size 1. Batch-specific
profiles, FP16 and the separable blur were not evaluated in this run.

The test set has been inspected during development. The parity
checks establish implementation agreement on these images, not
generalization to unseen production data.
