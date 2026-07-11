from pathlib import Path
import sys


def resource_path(relative_path):
    """Resolve a project asset in source runs and PyInstaller bundles."""
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return str(base_path / relative_path)


def source_path(relative_path):
    """Resolve a development-only source asset outside a frozen bundle."""
    return str(Path(__file__).resolve().parent / relative_path)

# deepcreampy 模型
deepcreampy_bar_model = resource_path("models/deepcreampy/bar.onnx")
deepcreampy_mosaic_model = resource_path("models/deepcreampy/mosaic.onnx")

# mrcnn模型
mrcnn_model = resource_path("models/mrcnn/weights.onnx")
mrcnn_source_model = source_path("models/mrcnn/weights.h5")

# esrgan 模型 (已从 PyTorch .pth 转为 ONNX)
esrgan_model = resource_path("models/esrgan/4x-Fatal-Pixels.onnx")
esrgan_model_data = resource_path("models/esrgan/4x-Fatal-Pixels.onnx.data")
