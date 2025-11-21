import cv2
from PIL import Image
import io
import numpy as np


def show_np_image(image_array):
    """
    将np数组图像显示
    :param image_array:
    :return:
    """
    # 如果图像是RGB格式，需要转换为BGR格式
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # 使用 OpenCV 显示图像
    cv2.imshow('Display Image', image_array)

    # 等待用户按键，然后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_cv_image(image):
    """
    显示cv2图像
    :param image:
    :return:
    """
    # 使用 OpenCV 显示图像
    cv2.imshow('Display Image', image)

    # 等待用户按键，然后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img2npimage(image):
    # 创建一个BytesIO对象，用于存储图片的二进制数据
    binary_data = io.BytesIO()
    image.save(binary_data,  format="PNG")
    # 获取二进制数据
    binary_data = binary_data.getvalue()
    return bytes2npimage(binary_data)

def bytes2npimage(image_bytes: bytes):
    """
    将二进制数据转成np数组
    :param image_bytes:
    :return:
    """

    image_bytes_io = io.BytesIO(image_bytes)
    # 读取字节流中的数据，并将其解码为图像
    # 注意第二个参数是颜色通道，对于大多数图像，它是3（代表RGB）
    image_array = cv2.imdecode(np.frombuffer(image_bytes_io.read(), np.uint8), cv2.IMREAD_COLOR)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    return image_array


def bytes2cvimage(image_bytes: bytes):
    """
    np矩阵转cv2矩阵
    :param image_bytes:
    :return:
    """
    image_bytes_io = io.BytesIO(image_bytes)
    # 读取字节流中的数据，并将其解码为图像
    # 注意第二个参数是颜色通道，对于大多数图像，它是3（代表RGB）
    image_array = cv2.imdecode(np.frombuffer(image_bytes_io.read(), np.uint8), cv2.IMREAD_COLOR)
    return image_array



def npimage2bytes(image_array):
    """
    np数组png文件
    :param image_array:
    :return:
    """
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return cvimage2bytes(image_array)


def cvimage2bytes(image_array):
    """
    cv2转png
    :param image_array:
    :return:
    """
    # 将数组转换为PNG图片的二进制数据
    _, binary_data = cv2.imencode('.png', image_array)
    return bytes(binary_data)


def del_alpha_channel(image_array):
    """
    删除图像的alpha通道
    :param image_array:
    :return:
    """
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    return image_array

