# mrcnn 实现自动涂抹
import cv2
import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from tools import image_tool
from tools.green_mask_project_mosaic_resolution import get_mosaic_res
from tools.decorators import timer_decorator
import os
import config

# 关闭一些tensorflow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}



class HentaiConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "hentai"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1

    # Number of training steps per epoch, equal to dataset train size
    STEPS_PER_EPOCH = 1490

    # Skip detections with < 75% confidence
    DETECTION_MIN_CONFIDENCE = 0.75


    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# 载入模型文件
model_path = config.mrcnn_model
log_path = "./logs"
model = modellib.MaskRCNN(mode="inference", config=HentaiConfig(), model_dir=log_path)
model.load_weights(model_path, by_name=True)

@timer_decorator
def detector(image_bytes: bytes, is_mosaic=False):
    image = image_tool.bytes2npimage(image_bytes)
    r = model.detect([image], verbose=0)[0]
    if len(r["scores"]) == 0:
        print("Skipping image with no detection")
        return  image_bytes

    if is_mosaic == True:
        remove_indices = np.where(r['class_ids'] != 2)  # remove bars: class 2
    else:
        remove_indices = np.where(r['class_ids'] != 1)  # remove mosaic: class 1
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
        mimg =cv2.erode(src=mask.astype('uint8'), kernel=kernel, iterations=1)  #
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