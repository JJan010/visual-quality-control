import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import tensorrt as trt

from visual_quality.inference.onnx_features import sha256_file
from visual_quality.inference.export_validation import load_validated_export


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if not 1 <= args.opt_batch <= 8:
        raise SystemExit("STOP: opt-batch musi należeć do zakresu 1-8.")
    if not torch.cuda.is_available():
        raise SystemExit("STOP: CUDA niedostępna.")

    torch.cuda.set_device(0)
    torch.cuda.init()

    run_dir = args.run_dir.resolve()
    export_dir = args.export_dir.resolve()
    onnx_path = export_dir / "feature_extractor.onnx"

    validation = load_validated_export(run_dir, export_dir)
    checkpoint_sha = validation["checkpoint_sha256"]
    onnx_sha = validation["onnx_sha256"]

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    if builder is None:
        raise RuntimeError("STOP: nie udało się utworzyć buildera.")

    network = builder.create_network(0)
    onnx_parser = trt.OnnxParser(network, logger)

    if not onnx_parser.parse_from_file(str(onnx_path)):
        errors = [
            str(onnx_parser.get_error(index))
            for index in range(onnx_parser.num_errors)
        ]
        raise RuntimeError("STOP: błędy parsera ONNX:\n" + "\n".join(errors))

    if network.num_inputs != 1 or network.num_outputs != 2:
        raise RuntimeError("STOP: nieoczekiwana liczba wejść lub wyjść.")

    input_tensor = network.get_input(0)
    if (
        input_tensor.name != "images"
        or tuple(input_tensor.shape) != (-1, 3, 256, 256)
        or input_tensor.dtype != trt.float32
    ):
        raise RuntimeError("STOP: nieoczekiwany kontrakt wejścia.")

    expected_outputs = {
        "layer2": (-1, 512, 32, 32),
        "layer3": (-1, 1024, 16, 16),
    }
    outputs = {
        network.get_output(index).name: network.get_output(index)
        for index in range(network.num_outputs)
    }
    if set(outputs) != set(expected_outputs):
        raise RuntimeError("STOP: nieoczekiwane nazwy wyjść.")

    for name, tensor in outputs.items():
        if (
            tuple(tensor.shape) != expected_outputs[name]
            or tensor.dtype != trt.float32
        ):
            raise RuntimeError(f"STOP: nieoczekiwany kontrakt {name}.")

    # Linear I/O layout will match contiguous PyTorch tensors.
    for tensor in [input_tensor, *outputs.values()]:
        tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)

    config = builder.create_builder_config()
    workspace_bytes = 1 << 30
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_bytes
    )

    # This first engine uses FP32 without TF32 or lower-precision modes.
    for flag in (
        trt.BuilderFlag.TF32,
        trt.BuilderFlag.FP16,
        trt.BuilderFlag.BF16,
        trt.BuilderFlag.INT8,
    ):
        config.clear_flag(flag)

    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    shapes = {
        "min": (1, 3, 256, 256),
        "opt": (args.opt_batch, 3, 256, 256),
        "max": (8, 3, 256, 256),
    }
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images", shapes["min"], shapes["opt"], shapes["max"]
    )

    if not bool(profile):
        raise RuntimeError("STOP: niepoprawny profil rozmiarów.")

    actual_shapes = [
        tuple(shape) for shape in profile.get_shape("images")
    ]
    expected_shapes = [
        shapes["min"], shapes["opt"], shapes["max"]
    ]
    if actual_shapes != expected_shapes:
        raise RuntimeError("STOP: profil ma inne rozmiary niż oczekiwane.")

    print("Profile shapes verified:", actual_shapes, flush=True)

    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("STOP: nie udało się dodać profilu.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (args.output_dir.resolve() if args.output_dir else
                  export_dir / f"tensorrt_fp32_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=False)

    build_config = {
        "checkpoint_sha256": checkpoint_sha,
        "onnx_sha256": onnx_sha,
        "build_script_sha256": sha256_file(Path(__file__)),
        "onnx_validation_sha256": sha256_file(export_dir / "pipeline_validation.json"),
        "tensorrt_version": trt.__version__,
        "torch_version": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "precision": "fp32_tf32_off",
        "tf32_enabled": config.get_flag(trt.BuilderFlag.TF32),
        "fp16_enabled": config.get_flag(trt.BuilderFlag.FP16),
        "bf16_enabled": config.get_flag(trt.BuilderFlag.BF16),
        "int8_enabled": config.get_flag(trt.BuilderFlag.INT8),
        "workspace_limit_bytes": workspace_bytes,
        "optimization_profile": shapes,
        "network_layers_before_build": network.num_layers,
    }
    save_json(output_dir / "build_config.json", build_config)

    print("GPU:", build_config["gpu"], flush=True)
    print("TensorRT:", trt.__version__, flush=True)
    print("Precision: FP32, TF32 disabled", flush=True)
    print("Optimization profile:", shapes, flush=True)
    print("Output directory:", output_dir, flush=True)
    print("Building TensorRT engine...", flush=True)

    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - start

    if serialized is None:
        raise RuntimeError("STOP: TensorRT nie zbudował silnika.")

    engine_path = output_dir / "feature_extractor.plan"
    engine_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("STOP: zapisany silnik nie daje się załadować.")

    if engine.num_io_tensors != 3 or engine.num_optimization_profiles != 1:
        raise RuntimeError("STOP: nieoczekiwana struktura silnika.")

    io_tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_dtype(name) != trt.float32:
            raise RuntimeError(f"STOP: {name} nie ma typu float32.")
        if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
            raise RuntimeError(f"STOP: {name} nie ma formatu LINEAR.")

        item = {
            "name": name,
            "mode": str(engine.get_tensor_mode(name)),
            "dtype": str(engine.get_tensor_dtype(name)),
            "shape": list(engine.get_tensor_shape(name)),
            "format": str(engine.get_tensor_format(name)),
        }
        io_tensors.append(item)
        print("Tensor:", item)

    report = {
        **build_config,
        "status": "built_and_deserialized",
        "engine_sha256": sha256_file(engine_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "build_seconds": build_seconds,
        "io_tensors": io_tensors,
        "numerical_validation": "pending",
        "performance_validation": "pending",
    }
    save_json(output_dir / "build_report.json", report)

    print(f"Build time: {build_seconds:.2f} s")
    print(f"Engine size: {engine_path.stat().st_size / 1024**2:.2f} MiB")
    print("Saved engine:", engine_path)
    print("TENSORRT BUILD: OK")


if __name__ == "__main__":
    main()
