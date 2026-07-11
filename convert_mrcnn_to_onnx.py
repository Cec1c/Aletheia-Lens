"""Convert the Matterport Mask R-CNN detector to an ONNX Runtime model."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import onnxruntime as ort
import tensorflow as tf

import config as project_config
from mrcnn.config import Config
from mrcnn import model as modellib


class HentaiConfig(Config):
    NAME = "hentai"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 3
    GPU_COUNT = 1
    DETECTION_MIN_CONFIDENCE = 0.75


def convert_mrcnn_to_onnx():
    output_path = Path("models/mrcnn/weights.onnx")
    temporary_output = output_path.with_name("weights.converted.onnx")
    backup_output = output_path.with_name("weights.previous.onnx")
    temporary_output.unlink(missing_ok=True)
    backup_output.unlink(missing_ok=True)

    print("[1/4] 构建 Mask R-CNN 并加载 H5 权重")
    model = modellib.MaskRCNN(
        mode="inference",
        config=HentaiConfig(),
        model_dir="./logs",
    )
    model.load_weights(project_config.mrcnn_source_model, by_name=True)

    print("[2/4] 导出临时 SavedModel")
    with tempfile.TemporaryDirectory(prefix="aletheia_mrcnn_") as saved_model_dir:
        tf.saved_model.save(model.keras_model, saved_model_dir)

        print("[3/4] 使用 tf2onnx 转换")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--saved-model",
                saved_model_dir,
                "--output",
                str(temporary_output),
                "--opset",
                "13",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            print(result.stdout[-2000:])
            print(result.stderr[-4000:])
            raise RuntimeError(f"tf2onnx failed with exit code {result.returncode}")

    print("[4/4] 验证 ONNX 可加载并识别合成马赛克")
    session = ort.InferenceSession(
        str(temporary_output),
        providers=["CPUExecutionProvider"],
    )
    print("inputs:", [value.name for value in session.get_inputs()])
    print("outputs:", [value.name for value in session.get_outputs()])

    if output_path.exists():
        os.replace(output_path, backup_output)
    os.replace(temporary_output, output_path)
    smoke_code = textwrap.dedent(
        """
        import numpy as np
        import detector

        image = np.full((512, 512, 3), 255, dtype=np.uint8)
        for y in range(160, 352, 16):
            for x in range(160, 352, 16):
                image[y:y + 16, x:x + 16] = 0 if ((x + y) // 16) % 2 == 0 else 128

        result = detector.detect_image(image)
        mosaic_indices = np.where(result['class_ids'] == 2)[0]
        assert mosaic_indices.size > 0
        best = mosaic_indices[np.argmax(result['scores'][mosaic_indices])]
        assert float(result['scores'][best]) > 0.9
        actual_mask = result['masks'][:, :, best].astype(bool)
        expected_mask = np.zeros((512, 512), dtype=bool)
        expected_mask[160:352, 160:352] = True
        mask_intersection = np.logical_and(actual_mask, expected_mask).sum()
        mask_union = np.logical_or(actual_mask, expected_mask).sum()
        mask_iou = mask_intersection / float(mask_union)
        assert mask_iou > 0.75
        """
    )
    try:
        smoke = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                smoke_code,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if smoke.returncode != 0:
            raise RuntimeError(
                "Converted model failed the synthetic mosaic smoke test:\n"
                + smoke.stdout[-2000:]
                + smoke.stderr[-4000:]
            )
    except Exception:
        output_path.unlink(missing_ok=True)
        if backup_output.exists():
            os.replace(backup_output, output_path)
        raise
    else:
        backup_output.unlink(missing_ok=True)

    print(f"转换完成: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    convert_mrcnn_to_onnx()
