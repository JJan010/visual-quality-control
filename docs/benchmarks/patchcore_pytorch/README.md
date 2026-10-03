# PatchCore — initial PyTorch inference benchmark

## Correctness check

The reusable PatchcorePredictor was checked against the existing
evaluation on all 83 test images:

- Prepared image tensors matched exactly.
- Maximum absolute image-score difference: 0.
- Changed classification decisions: 0.
- Anomaly-map shapes and finite values were verified.
- Pixel-wise map equality was not tested.

See predictor_parity.json for the checkpoint identity and tolerances.

## Measurement scope

Start: a prepared float32 RGB batch in pageable CPU memory.
End: image scores, decisions, and anomaly maps available in CPU memory.

Included:
- CPU-to-GPU transfer.
- ImageNet normalization.
- Feature extraction.
- Memory-bank search and anomaly scoring.
- Anomaly-map generation and threshold comparison.
- GPU-to-CPU transfer of all outputs.

Excluded:
- Model loading.
- File reading, decoding, resizing, and tensor conversion.
- Batch assembly.
- HTTP, queueing, and serialization.

## Setup

- GPU: NVIDIA GeForce RTX 5080.
- Input: 256 x 256.
- Memory bank: 1710 x 1536.
- Float32 tensors, no autocast.
- 30 warmup calls and 100 measured calls per batch size.
- Sequential execution with CUDA synchronization.
- Prepared test images reused cyclically.
- Incomplete batches filled with images from the start of the list.

Detailed runtime settings and input batches are stored in benchmark.json.

## Results

| Metric | Batch 1 | Batch 8 |
|---|---:|---:|
| Mean batch latency, ms | 5.335 | 18.600 |
| p50 batch latency, ms | 4.243 | 18.787 |
| p95 batch latency, ms | 8.960 | 19.168 |
| Serial images per second | 187.46 | 430.11 |
| Peak allocated memory, MiB | 143.90 | 242.15 |
| Peak reserved memory, MiB | 164.00 | 394.00 |

Throughput is total processed images divided by the sum of measured
call durations. It does not include time spent waiting to assemble a batch.

Batch 8 achieved approximately 2.29 times the throughput of batch 1,
with higher batch latency and GPU memory usage.

## Limitations

This is an initial measurement from one process and a short run.
It is not a sustained-load or end-to-end service benchmark.

PyTorch memory statistics do not represent total device memory usage.
The cause of the wider batch-1 latency distribution has not been established.

## Next step

Profile feature extraction and memory-bank search separately to identify
which component should be prioritized for optimization.
