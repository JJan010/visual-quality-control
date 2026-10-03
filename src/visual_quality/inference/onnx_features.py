import hashlib
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from torch import nn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OnnxCudaFeatureExtractor(nn.Module):
    """PatchCore feature extractor with GPU inputs and outputs.

    Contract:
        - normalized float32 input: (B, 3, 256, 256);
        - batch size: 1-8;
        - CUDA device and stream fixed during initialization;
        - sequential inference calls.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        expected_sha256: str,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()

        self.onnx_path = Path(onnx_path).resolve()
        self.onnx_sha256 = sha256_file(self.onnx_path)
        if self.onnx_sha256 != expected_sha256:
            raise RuntimeError("ONNX file does not match the validated export.")

        requested = torch.device(device)
        if requested.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This adapter requires CUDA.")

        index = requested.index
        if index is None:
            index = torch.cuda.current_device()
        self.device = torch.device("cuda", index)
        self.stream = torch.cuda.current_stream(self.device)

        ort.preload_dlls(cuda=True, cudnn=True, msvc=False)

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": index,
                        "use_tf32": 0,
                        "user_compute_stream": str(self.stream.cuda_stream),
                    },
                )
            ],
        )
        self.session.disable_fallback()

        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("CUDA Execution Provider was not initialized.")

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if len(inputs) != 1 or inputs[0].name != "images":
            raise RuntimeError("Unexpected ONNX inputs.")
        if inputs[0].type != "tensor(float)":
            raise RuntimeError("Expected float32 ONNX input.")
        if inputs[0].shape[1:] != [3, 256, 256]:
            raise RuntimeError("Unexpected ONNX input shape.")

        if [item.name for item in outputs] != ["layer2", "layer3"]:
            raise RuntimeError("Unexpected ONNX outputs.")
        for item, shape in zip(
            outputs, ([512, 32, 32], [1024, 16, 16])
        ):
            if item.type != "tensor(float)" or item.shape[1:] != shape:
                raise RuntimeError(f"Unexpected output contract: {item.name}")

        self.eval()

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.device != self.device:
            raise ValueError(f"Expected input on {self.device}.")
        if images.dtype != torch.float32:
            raise ValueError("Expected float32 input.")
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 256, 256):
            raise ValueError("Expected input shape (B, 3, 256, 256).")
        if not 1 <= images.shape[0] <= 8:
            raise ValueError("Supported batch size: 1-8.")

        current = torch.cuda.current_stream(self.device)
        if current.cuda_stream != self.stream.cuda_stream:
            raise RuntimeError("Use the CUDA stream selected at initialization.")

        images = images.contiguous()
        batch_size = images.shape[0]

        # PyTorch owns the output memory; ORT writes into these tensors.
        outputs = {
            "layer2": torch.empty(
                (batch_size, 512, 32, 32),
                device=self.device,
                dtype=torch.float32,
            ),
            "layer3": torch.empty(
                (batch_size, 1024, 16, 16),
                device=self.device,
                dtype=torch.float32,
            ),
        }

        binding = self.session.io_binding()
        binding.bind_input(
            name="images",
            device_type="cuda",
            device_id=self.device.index,
            element_type=np.float32,
            shape=tuple(images.shape),
            buffer_ptr=images.data_ptr(),
        )

        for name, tensor in outputs.items():
            binding.bind_output(
                name=name,
                device_type="cuda",
                device_id=self.device.index,
                element_type=np.float32,
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )

        self.session.run_with_iobinding(binding)
        return outputs
