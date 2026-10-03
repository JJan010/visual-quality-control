import json
from importlib.metadata import version
from pathlib import Path

# Importujemy PyTorch przed utworzeniem sesji ONNX Runtime.
import torch
import torch.nn.functional as F

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "artifacts/environment/onnx_gpu_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("PyTorch:", torch.__version__)
    print("CUDA w PyTorch:", torch.version.cuda)
    print("ONNX:", version("onnx"))
    print("ONNX Script:", version("onnxscript"))
    print("ONNX Runtime:", ort.__version__)

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch nie widzi GPU.")

    print("GPU:", torch.cuda.get_device_name(0))
    ort.preload_dlls(cuda=True, cudnn=True, msvc=False)

    available = ort.get_available_providers()
    print("Available providers:", available)
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError("ONNX Runtime nie udostępnia CUDA.")

    rng = np.random.default_rng(42)
    images = rng.standard_normal((2, 3, 32, 32)).astype(np.float32)
    weights = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)

    # Mały graf: wejście + zapisane wagi -> konwolucja -> wyjście.
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Conv",
                inputs=["images", "weights"],
                outputs=["output"],
                name="test_conv",
                kernel_shape=[3, 3],
            )
        ],
        name="cuda_conv_check",
        inputs=[
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [2, 3, 32, 32]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [2, 4, 30, 30]
            )
        ],
        initializer=[numpy_helper.from_array(weights, name="weights")],
    )

    # Jawna wersja formatu grafu, obsługiwana przez wybrany runtime.
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.checker.check_model(model)

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(output_dir / "ort_profile")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0})
        ],
    )
    session.disable_fallback()

    print("Session providers:", session.get_providers())
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("Sesja nie uruchomiła dostawcy CUDA.")

    actual = session.run(["output"], {"images": images})[0]
    profile_path = Path(session.end_profiling())

    # Referencję liczymy na CPU; badanym wykonaniem GPU jest ONNX Runtime.
    with torch.inference_mode():
        expected = F.conv2d(
            torch.from_numpy(images),
            torch.from_numpy(weights),
        ).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    events = json.loads(profile_path.read_text(encoding="utf-8"))
    conv_providers = {
        event.get("args", {}).get("provider")
        for event in events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("op_name") == "Conv"
        and event.get("args", {}).get("provider")
    }

    print("Conv execution providers:", sorted(conv_providers))
    if conv_providers != {"CUDAExecutionProvider"}:
        raise RuntimeError("Profil nie potwierdza wykonania Conv na GPU.")

    max_error = float(np.max(np.abs(actual - expected)))
    print("Output shape:", actual.shape)
    print(f"Maximum absolute difference: {max_error:.10f}")
    print("Profile:", profile_path)
    print("ONNX GPU CHECK: OK")


if __name__ == "__main__":
    main()
