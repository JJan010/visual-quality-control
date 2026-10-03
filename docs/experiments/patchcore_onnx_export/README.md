# PatchCore feature extractor: ONNX export and numerical validation

## Scope

The checkpoint's Wide ResNet-50-2 feature extractor was exported to
ONNX with opset 18 and IR version 10. It accepts normalized float32
images with shape (B, 3, 256, 256) and returns layer2 and layer3.

The intended batch range is 1-8. Feature diagnostics exercised
batch sizes 1 and 8. Normalization and PatchCore postprocessing
remain outside the ONNX graph.

## Initial feature check

The initial elementwise comparison used atol=1e-4 and rtol=1e-4.
It failed for a small fraction of feature values.

Further diagnostics also found tolerance violations between
PyTorch CPU and PyTorch CUDA. The original feature check remains
recorded as failed; its tolerance was not increased.

ONNX Runtime profiling confirmed CUDA execution of convolutions.

## Pipeline comparison

The same checkpoint, memory bank, threshold and original_2d blur
were used for both implementations.

The first comparison preserved the baseline's TF32-enabled PyTorch
convolution setting, while ONNX Runtime had TF32 disabled.
Its maximum score difference was 0.05853653. Decisions were unchanged,
but score and map differences exceeded the original tolerance.

A controlled comparison then used IEEE FP32 settings in PyTorch
and use_tf32=0 in ONNX Runtime:

- Images: 83, batch size 8, final batch size 3.
- Threshold: 32.44375991821289, unchanged.
- Maximum score difference: 0.0001354218.
- Scores outside tolerance: 0.
- Maximum map difference: 0.0003976822.
- Mean map difference: 0.0000199493.
- Map pixels outside tolerance: 0 of 5,439,488.
- Changed decisions between implementations: 0.
- Changed decisions relative to the saved baseline: 0 for both.

The elementwise criterion remained:
abs(candidate - reference) <= 1e-4 + 1e-4 * abs(reference).

These results support pipeline parity for the tested configuration.
The previous precision settings explain the dominant part of the
observed discrepancy.

## Limitations and next step

The test set has been inspected during development. This comparison
checks implementation parity, not generalization to unseen data.

The diagnostic adapter transfers features through CPU memory.
No performance claim is made. A GPU I/O Binding adapter is the next
implementation step and requires its own parity check.

Full-pipeline parity for batch size 1, the separable blur backend,
and TensorRT has not been established by this experiment.

Raw diagnostic reports retain their original status.
validation_summary.json records the scoped review outcome.
