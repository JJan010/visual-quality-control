import json
from pathlib import Path

import torch
import tensorrt as trt
from torch import nn

from visual_quality.inference.onnx_features import sha256_file


class TensorRTCudaFeatureExtractor(nn.Module):
    """Sequential FP32 TensorRT inference with GPU inputs and outputs."""

    def __init__(self, engine_dir: str | Path) -> None:
        super().__init__()
        self.device = torch.device("cuda:0")
        self.engine_dir = Path(engine_dir).resolve()
        self.report = json.loads(
            (self.engine_dir / "build_report.json").read_text(
                encoding="utf-8"
            )
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        if torch.cuda.current_device() != 0:
            raise RuntimeError("This adapter expects CUDA device 0.")
        if self.report["status"] != "built_and_deserialized":
            raise RuntimeError("Engine build was not completed.")
        if self.report["precision"] != "fp32_tf32_off":
            raise RuntimeError("Expected the FP32 engine.")
        if any(self.report[key] for key in (
            "tf32_enabled", "fp16_enabled",
            "bf16_enabled", "int8_enabled",
        )):
            raise RuntimeError("Unexpected engine precision settings.")
        if self.report["tensorrt_version"] != trt.__version__:
            raise RuntimeError("TensorRT version differs from the build.")
        if self.report["gpu"] != torch.cuda.get_device_name(0):
            raise RuntimeError("GPU differs from the build.")

        engine_path = self.engine_dir / "feature_extractor.plan"
        self.engine_sha256 = sha256_file(engine_path)
        if self.engine_sha256 != self.report["engine_sha256"]:
            raise RuntimeError("Engine file does not match its report.")

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            engine_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError("Could not deserialize the engine.")

        expected = {
            "images": (trt.TensorIOMode.INPUT, (-1, 3, 256, 256)),
            "layer2": (trt.TensorIOMode.OUTPUT, (-1, 512, 32, 32)),
            "layer3": (trt.TensorIOMode.OUTPUT, (-1, 1024, 16, 16)),
        }
        names = {
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        }
        if names != set(expected):
            raise RuntimeError("Unexpected engine I/O names.")
        if self.engine.num_optimization_profiles != 1:
            raise RuntimeError("Expected one optimization profile.")

        for name, (mode, shape) in expected.items():
            if (
                self.engine.get_tensor_mode(name) != mode
                or tuple(self.engine.get_tensor_shape(name)) != shape
                or self.engine.get_tensor_dtype(name) != trt.float32
                or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
                or self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
            ):
                raise RuntimeError(f"Unexpected tensor contract: {name}")

        profile = [
            tuple(shape)
            for shape in self.engine.get_tensor_profile_shape("images", 0)
        ]
        if profile[0] != (1, 3, 256, 256) or profile[2] != (8, 3, 256, 256):
            raise RuntimeError("Unexpected supported batch range.")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create an execution context.")

        # Preserve detailed engine metadata without runtime NVTX overhead.
        self.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
        self.stream = torch.cuda.Stream(device=self.device)

        if not self.context.set_optimization_profile_async(
            0, self.stream.cuda_stream
        ):
            raise RuntimeError("Could not select the optimization profile.")
        self.stream.synchronize()
        self.eval()

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.device != self.device or images.dtype != torch.float32:
            raise ValueError("Expected float32 input on cuda:0.")
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
            raise ValueError("Expected shape (B, 3, 256, 256).")
        if not 1 <= images.shape[0] <= 8:
            raise ValueError("Supported batch size: 1-8.")

        images = images.contiguous()
        batch_size = images.shape[0]

        if not self.context.set_input_shape("images", tuple(images.shape)):
            raise RuntimeError("TensorRT rejected the input shape.")

        outputs = {}
        for name, expected_shape in (
            ("layer2", (batch_size, 512, 32, 32)),
            ("layer3", (batch_size, 1024, 16, 16)),
        ):
            shape = tuple(self.context.get_tensor_shape(name))
            if shape != expected_shape:
                raise RuntimeError(f"Unexpected runtime shape: {name}")

            outputs[name] = torch.empty(
                shape, device=self.device, dtype=torch.float32
            )

        for name, tensor in {"images": images, **outputs}.items():
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Could not bind tensor: {name}")

        # Input preparation may still be running in the caller's stream.
        self.stream.wait_stream(torch.cuda.current_stream(self.device))

        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed.")

        # Keep buffers alive until execution completes.
        self.stream.synchronize()
        return outputs
