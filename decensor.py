"""
Cleaned from: DeepCreamPy/decensor.py
"""

import sys
from multiprocessing.pool import ThreadPool
import numpy as np
from PIL import Image
from tools import image_tool
from scipy.ndimage import measurements
from predict import predict
from deepcreampy.utils import image_to_array, expand_bounding
from tools.decorators import timer_decorator

# Green.
MASK_COLOR = [0, 1, 0]


def find_mask(colored):
    mask = np.ones(colored.shape, np.uint8)
    i, j = np.where(np.all(colored[0] == MASK_COLOR, axis=-1))
    mask[0, i, j] = 0
    return mask


# Performant connected-components algorithm.
def find_regions(image, mask_color):
    pixels = np.array(image)
    array = np.all(pixels == mask_color, axis=2)
    labeled, n_components = measurements.label(array)
    indices = np.moveaxis(np.indices(array.shape), 0, -1)[:, :, [1, 0]]

    regions = []
    for index in range(1, n_components + 1):
        regions.append(indices[labeled == index].tolist())
    regions.sort(key=len, reverse=True)
    return regions


@timer_decorator
def decensor(ori: Image, colored: Image, is_mosaic: bool):
    # save the alpha channel if the image has an alpha channel
    has_alpha = False
    if ori.mode == "RGBA":
        has_alpha = True
        alpha_channel = np.asarray(ori)[:, :, 3]
        alpha_channel = np.expand_dims(alpha_channel, axis=-1)
        ori = ori.convert('RGB')

    ori_array = image_to_array(ori)
    if ori_array.ndim != 3:
        print("输入图像维度不正确, 可能是一张灰度图")
        return ori
    if is_mosaic:
        # if mosaic decensor, mask is empty
        colored = colored.convert('RGB')
        color_array = image_to_array(colored)
        color_array = np.expand_dims(color_array, axis=0)
        mask = find_mask(color_array)
    else:
        ori_array_mask = np.expand_dims(ori_array, axis=0)
        mask = find_mask(ori_array_mask)

    # colored image is only used for finding the regions
    regions = find_regions(colored.convert('RGB'), [v * 255 for v in MASK_COLOR])
    print("Found {region_count} censored regions in this image!".format(region_count=len(regions)))
    if len(regions) == 0 and not is_mosaic:
        print("No green (0,255,0) regions detected! Make sure you're using exactly the right color.")
        return ori

    def predict_region(region):
        bounding_box = expand_bounding(ori, region, expand_factor=1.5)
        crop_img = ori.crop(bounding_box)

        # convert mask back to image
        mask_reshaped = mask[0, :, :, :] * 255.0
        mask_img = Image.fromarray(mask_reshaped.astype('uint8'))
        # resize the cropped images

        crop_img = crop_img.resize((256, 256), resample=Image.NEAREST)
        crop_img_array = image_to_array(crop_img)

        # resize the mask images
        mask_img = mask_img.crop(bounding_box)
        mask_img = mask_img.resize((256, 256), resample=Image.NEAREST)

        # convert mask_img back to array
        mask_array = image_to_array(mask_img)
        # the mask has been upscaled so there will be values not equal to 0 or 1

        if not is_mosaic:
            a, b = np.where(np.all(mask_array == 0, axis=-1))
            crop_img_array[a, b, :] = 0.

        # Normalize.
        crop_img_array = crop_img_array * 2.0 - 1

        # Queue prediction request.
        pred_img_array = predict(crop_img_array, mask_array, is_mosaic)
        pred_img_array = (255.0 * ((pred_img_array + 1.0) / 2.0)).astype(np.uint8)
        return pred_img_array, bounding_box

    # Run predictions.
    with ThreadPool() as pool:
        results = pool.map(predict_region, regions)

    output_img_array = ori_array.copy()
    for (pred_img_array, bounding_box), region in zip(results, regions):
        # scale prediction image back to original size
        bounding_width = bounding_box[2] - bounding_box[0]
        bounding_height = bounding_box[3] - bounding_box[1]

        # convert np array to image
        pred_img = Image.fromarray(pred_img_array.astype('uint8'))
        pred_img = pred_img.resize((bounding_width, bounding_height), resample=Image.BICUBIC)
        pred_img_array = image_to_array(pred_img)

        # Efficiently copy regions into output image.
        for (x, y) in region:
            if bounding_box[0] <= x < bounding_box[0] + bounding_width:
                if bounding_box[1] <= y < bounding_box[1] + bounding_height:
                    output_img_array[y][x] = pred_img_array[y - bounding_box[1]][x - bounding_box[0]]

    output_img_array = output_img_array * 255.0

    # restore the alpha channel if the image had one
    if has_alpha:
        output_img_array = np.concatenate((output_img_array, alpha_channel), axis=2)

    print("Decensored image. Returning it.")
    return Image.fromarray(output_img_array.astype('uint8'))
