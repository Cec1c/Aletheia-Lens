"""Optional DSP preprocessing for printed manga screentones.

The filter follows the common Gaussian blur, bilateral filter, and
Laplacian-style sharpening approach described by natethegreate's
Screentone-Remover project, while keeping this implementation independent
and preserving the input alpha channel exactly.
"""

import cv2
import numpy as np
from PIL import Image


_LEVEL_CONFIG = {
    1: {"gaussian_kernel": 5, "bilateral_sigma_color": 30},
    2: {"gaussian_kernel": 5, "bilateral_sigma_color": 50},
    3: {"gaussian_kernel": 7, "bilateral_sigma_color": 80},
}

_SHARPEN_KERNEL = np.array(
    [
        [0.0, -1.14, 0.0],
        [-1.14, 5.56, -1.14],
        [0.0, -1.14, 0.0],
    ],
    dtype=np.float32,
)


def remove_screentones(image: Image.Image, level: int = 2) -> Image.Image:
    """Remove high-frequency screentones while retaining major edges.

    Args:
        image: Source PIL image. RGB and transparency are both supported.
        level: Filter strength from 1 (light) through 3 (strong).

    Returns:
        A new RGB or RGBA PIL image with the same dimensions as ``image``.
    """
    if level not in _LEVEL_CONFIG:
        raise ValueError(f"去网点强度必须是 1、2 或 3，当前值: {level}")

    has_alpha = "A" in image.getbands() or "transparency" in image.info
    working = image.convert("RGBA" if has_alpha else "RGB")
    image_array = np.asarray(working, dtype=np.uint8)
    rgb = image_array[:, :, :3]
    alpha = image_array[:, :, 3].copy() if has_alpha else None

    config = _LEVEL_CONFIG[level]
    kernel_size = config["gaussian_kernel"]
    blurred = cv2.GaussianBlur(rgb, (kernel_size, kernel_size), 0)
    smoothed = cv2.bilateralFilter(
        blurred,
        d=7,
        sigmaColor=config["bilateral_sigma_color"],
        sigmaSpace=80,
    )
    filtered_rgb = cv2.filter2D(smoothed, -1, _SHARPEN_KERNEL)

    if alpha is None:
        return Image.fromarray(filtered_rgb)

    filtered_rgba = np.dstack((filtered_rgb, alpha))
    return Image.fromarray(filtered_rgba)
