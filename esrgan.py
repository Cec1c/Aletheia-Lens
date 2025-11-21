# esrgan 超分辨率
import cv2
import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from tools import image_tool
from tools.green_mask_project_mosaic_resolution import get_mosaic_res
from tools.decorators import timer_decorator
from ColabESRGAN.esrgan import EsrGan
import detector
import os
import config



model = EsrGan(config.esrgan_model)


@timer_decorator
def esrgan(image_bytes: bytes):
    image = image_tool.bytes2npimage(image_bytes)       # 读取图像并转成cv2的数组
    image_gan = get_gan_img(image)
    masks = get_masks(image)
    image = splice_masks(image, image_gan, masks)
    cov, mask = detector.apply_cover(image, masks, 0)
    return image_tool.npimage2bytes(image), image_tool.npimage2bytes(cov)



def get_gan_img(image):
    """
    将图像缩小， 并使用esrgan放大， 以此来清理马赛克
    :param image:
    :return:
    """
    image_r = image.copy()
    image_r = image_tool.del_alpha_channel(image_r)  # 删除图像的alpha通道
    granularity = get_mosaic_res(image_r)
    if granularity < 10:  # 图像缩放倍率
        print("Granularity of image was less than threshold at ", granularity)
        granularity = 10
    mini_img = cv2.resize(image_r, (int(image_r.shape[1] / granularity), int(image_r.shape[0] / granularity)),
                      interpolation=cv2.INTER_AREA)

    image_gan = model.run_esrgan(mini_img, mosaic_res=granularity)
    image_gan = cv2.resize(image_gan, (image.shape[1], image.shape[0])) # 将gan图像转成和原图同样大小
    return image_gan

@timer_decorator
def get_masks(image):
    """
    使用mrcnn推理出原图需要修复马赛克的位置
    :return:
    """
    image_r = image.copy()
    image_r = image_tool.del_alpha_channel(image_r)  # 删除图像的alpha通道
    r = detector.model.detect([image_r], verbose=0)[0]
    remove_indices = np.where(r['class_ids'] != 2)
    new_masks = np.delete(r['masks'], remove_indices, axis=2)
    return new_masks


def splice_masks(image, gan_image, mask):
    if mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) < 1)
        mask = np.repeat(mask, 3, axis=-1)  # 重新将mask改成和原图相同形状
        result = np.where(mask, image, gan_image)   # 如果mask的对应点为true， 使用gan的数据， 否则使用image的数据
        return result
    else:
        return image

