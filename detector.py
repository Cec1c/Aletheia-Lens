# mrcnn 实现自动涂抹 (ONNX Runtime 版本, 零 TensorFlow 依赖)
# 所有 mrcnn 工具函数内联，避免 from mrcnn.utils import (会触发 TF 导入)
import cv2
import numpy as np
import os
from tools import image_tool
from tools.decorators import timer_decorator
import config
from mrcnn.config import Config  # 纯 Python, 不依赖 TF
from onnx_runtime import create_session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# MaskRCNN 配置类
# ============================================================
class HentaiConfig(Config):
    NAME = "hentai"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 + 1  # BG + bar + mosaic
    STEPS_PER_EPOCH = 1490
    DETECTION_MIN_CONFIDENCE = 0.75
    GPU_COUNT = 1

_cfg = HentaiConfig()

# ============================================================
# ONNX 推理会话
# ============================================================
_onnx_session = create_session(config.mrcnn_model)

# ============================================================
# 内联的 mrcnn 工具函数 (纯 numpy/cv2，零 TF/skimage 依赖)
# ============================================================

def _resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """保持宽高比缩放图像 (来自 mrcnn.utils.resize_image)"""
    image_dtype = image.dtype
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    if min_dim:
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    if scale != 1.0:
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))))

    if mode == "square":
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        top_pad = bottom_pad = left_pad = right_pad = 0
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode="constant", constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def _mold_image(images, config):
    """RGB 图像减去均值像素 (来自 mrcnn.model.mold_image)"""
    return images.astype(np.float32) - config.MEAN_PIXEL


def _compose_image_meta(image_id, original_image_shape, image_shape, window, scale, active_class_ids):
    """组装图像元数据为 1D 数组 (来自 mrcnn.model.compose_image_meta)"""
    meta = np.array(
        [image_id]
        + list(original_image_shape)
        + list(image_shape)
        + list(window)
        + [scale]
        + list(active_class_ids)
    )
    return meta


def _generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """生成 RPN anchors (来自 mrcnn.utils.generate_anchors)"""
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    boxes = np.concatenate(
        [box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1
    )
    return boxes


def _generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """在特征金字塔各级生成 anchors (来自 mrcnn.utils.generate_pyramid_anchors)"""
    anchors = []
    for i in range(len(scales)):
        anchors.append(
            _generate_anchors(scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_stride)
        )
    return np.concatenate(anchors, axis=0)


def _norm_boxes(boxes, shape):
    """像素坐标 → 归一化坐标 (来自 mrcnn.utils.norm_boxes)"""
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def _denorm_boxes(boxes, shape):
    """归一化坐标 → 像素坐标 (来自 mrcnn.utils.denorm_boxes)"""
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def _unmold_mask(mask, bbox, image_shape):
    """将神经网络生成的 mask 还原为原始图像尺寸的二进制 mask (使用 cv2.resize 替代 skimage)"""
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    h, w = y2 - y1, x2 - x1
    mask = cv2.resize(mask.astype(np.float32), (max(1, w), max(1, h)))
    if mask.ndim == 0:
        mask = np.full((max(1, h), max(1, w)), mask, dtype=np.float32)
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool_)

    full_mask = np.zeros(image_shape[:2], dtype=np.bool_)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


def _compute_backbone_shapes(image_shape):
    """计算特征金字塔各级形状 (来自 mrcnn.model.compute_backbone_shapes)"""
    return np.array(
        [[int(np.ceil(image_shape[0] / stride)),
          int(np.ceil(image_shape[1] / stride))]
         for stride in _cfg.BACKBONE_STRIDES]
    )


# Anchor 缓存
_anchor_cache = {}

def _get_anchors(image_shape):
    """获取给定图像尺寸的 anchor 金字塔（带缓存）"""
    key = tuple(image_shape[:2])
    if key not in _anchor_cache:
        backbone_shapes = _compute_backbone_shapes(image_shape)
        a = _generate_pyramid_anchors(
            _cfg.RPN_ANCHOR_SCALES,
            _cfg.RPN_ANCHOR_RATIOS,
            backbone_shapes,
            _cfg.BACKBONE_STRIDES,
            _cfg.RPN_ANCHOR_STRIDE,
        )
        _anchor_cache[key] = _norm_boxes(a, image_shape[:2])
    return _anchor_cache[key]


def _mold_inputs(images):
    """预处理图像列表：缩放、归一化、生成元数据"""
    molded_images, image_metas, windows = [], [], []
    for image in images:
        molded_image, window, scale, padding, crop = _resize_image(
            image,
            min_dim=_cfg.IMAGE_MIN_DIM,
            min_scale=_cfg.IMAGE_MIN_SCALE,
            max_dim=_cfg.IMAGE_MAX_DIM,
            mode=_cfg.IMAGE_RESIZE_MODE,
        )
        molded_image = _mold_image(molded_image, _cfg)
        image_meta = _compose_image_meta(
            0, image.shape, molded_image.shape, window, scale,
            np.zeros([_cfg.NUM_CLASSES], dtype=np.int32),
        )
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
    return np.stack(molded_images), np.stack(image_metas), np.stack(windows)


def _unmold_detections(detections, mrcnn_mask, original_image_shape, image_shape, window):
    """后处理：将 ONNX 模型输出转为像素坐标检测结果"""
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]

    window = _norm_boxes(window, image_shape[:2])
    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh, ww = wy2 - wy1, wx2 - wx1
    scale = np.array([wh, ww, wh, ww])
    boxes = np.divide(boxes - shift, scale)
    boxes = _denorm_boxes(boxes, original_image_shape[:2])

    exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    full_masks = []
    for i in range(N):
        full_masks.append(_unmold_mask(masks[i], boxes[i], original_image_shape))
    full_masks = (
        np.stack(full_masks, axis=-1)
        if full_masks
        else np.empty(original_image_shape[:2] + (0,), dtype=np.bool_)
    )

    return boxes, class_ids, scores, full_masks


def _detect_onnx(image):
    """
    使用 ONNX Runtime 执行 MaskRCNN 检测（替代原先的 TF model.detect()）。
    输入: 单张 numpy 图像 (H, W, 3) uint8 RGB
    输出: dict {'rois', 'class_ids', 'scores', 'masks'}
    """
    molded_images, image_metas, windows = _mold_inputs([image])
    image_shape = molded_images[0].shape
    anchors = _get_anchors(image_shape)
    anchors = np.broadcast_to(anchors, (1,) + anchors.shape)

    # ONNX 推理
    onnx_inputs = {
        'input_image': molded_images.astype(np.float32),
        'input_image_meta': image_metas.astype(np.float32),
        'input_anchors': anchors.astype(np.float32),
    }
    detections, mrcnn_mask = _onnx_session.run(
        ['mrcnn_detection', 'mrcnn_mask'],
        onnx_inputs,
    )

    final_rois, final_class_ids, final_scores, final_masks = _unmold_detections(
        detections[0], mrcnn_mask[0],
        image.shape, molded_images[0].shape, windows[0],
    )
    return {
        'rois': final_rois,
        'class_ids': final_class_ids,
        'scores': final_scores,
        'masks': final_masks,
    }


def detect_image(image):
    """Public image-array interface shared by modes II and III."""
    return _detect_onnx(image)


# ============================================================
# 对外接口：与原版 detector 完全兼容
# ============================================================

@timer_decorator
def detector(image_bytes: bytes, is_mosaic=False):
    image = image_tool.bytes2npimage(image_bytes)
    r = detect_image(image)

    if len(r["scores"]) == 0:
        print("Skipping image with no detection")
        return image_bytes

    if is_mosaic:
        remove_indices = np.where(r['class_ids'] != 2)  # mosaic: 保留 class 2
    else:
        remove_indices = np.where(r['class_ids'] != 1)  # bar: 保留 class 1
    new_masks = np.delete(r['masks'], remove_indices, axis=2)
    cov, mask = apply_cover(image, new_masks, 0)
    image = image_tool.npimage2bytes(cov)
    return image


def apply_cover(image, mask, dilation):
    # Copy color pixels from the original color image where mask is set
    green = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
    green[:, :] = [0, 255, 0]

    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) < 1)
        # dilate mask to ensure proper coverage
        mimg = mask.astype('uint8') * 255
        kernel = np.ones((dilation, dilation), np.uint8)
        mimg = cv2.erode(src=mask.astype('uint8'), kernel=kernel, iterations=1)
        # dilation returns image with channels stripped (?!?). Reconstruct image channels
        mask_img = np.zeros([mask.shape[0], mask.shape[1], 3]).astype('bool')
        mask_img[:, :, 0] = mimg.astype('bool')
        mask_img[:, :, 1] = mimg.astype('bool')
        mask_img[:, :, 2] = mimg.astype('bool')

        cover = np.where(mask_img.astype('bool'), image, green).astype(np.uint8)
    else:
        # error case, return image
        cover = image
    return cover, mask
