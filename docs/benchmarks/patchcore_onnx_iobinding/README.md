# PatchCore: PyTorch vs ONNX Runtime with CUDA I/O Binding

## Correctness

The I/O Binding adapter accepts and returns CUDA tensors.
Its feature outputs matched ordinary ONNX Runtime execution exactly
for the tested validation inputs at batch sizes 1, 3 and 8.

Full-pipeline comparisons covered all 83 test images separately
at batch sizes 1 and 8:

- PyTorch IEEE FP32; ONNX Runtime use_tf32=0.
- Original 2D Gaussian blur in both implementations.
- Scores and maps passed atol=1e-4, rtol=1e-4.
- No changed decisions between implementations.
- No changed decisions relative to the saved baseline.
- The checkpoint, memory bank and threshold were unchanged.

See iobinding_parity.json for exact results and artifact hashes.

## Benchmark method

Hardware: NVIDIA GeForce RTX 5080.

Each variant received 30 warmup calls and 100 measured calls per
batch size. Measurements were paired on identical inputs, with
alternating execution order.

Scope: prepared CPU RGB batch, transfer to GPU, normalization,
complete PatchCore inference, and scores, labels and maps on CPU.

Disk reads, image decoding, resizing, CPU batch creation and model
loading were excluded. CUDA was synchronized at timing boundaries.

Only the feature extractor backend changed. Nearest-neighbor
search and anomaly-map generation remained in PyTorch.

## Results

| Batch | Backend | Mean ms | p50 ms | p95 ms | Serial images/s |
|---|---|---:|---:|---:|---:|
| 1 | PyTorch | 6.609 | 6.279 | 7.841 | 151.30 |
| 1 | ONNX Runtime I/O Binding | 6.391 | 6.181 | 7.265 | 156.47 |
| 8 | PyTorch | 23.496 | 23.595 | 24.122 | 340.48 |
| 8 | ONNX Runtime I/O Binding | 23.476 | 23.562 | 24.300 | 340.78 |

Mean latency ratios: 1.034x at batch 1 and 1.001x at batch 8.

## Interpretation

This run showed a small mean-latency advantage at batch 1 and
essentially equal performance at batch 8. It does not establish
a repeatable practical speedup.

The benchmark measures the complete pipeline and does not isolate
feature-extractor latency. These IEEE FP32 results should not be
directly compared with earlier TF32-enabled benchmark runs.

The validated ONNX artifact and GPU adapter provide a reference
for the next TensorRT experiment. TensorRT was not used here.
