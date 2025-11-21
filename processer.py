import io
import argparse
import sys
from PIL import Image
from detector import detector, apply_cover
from decensor import decensor
from esrgan import esrgan


def process_bar_auto(image_bytes: bytes) -> Image.Image:
    """
    处理色条自动修复模式（模式1）
    :param image_bytes: 输入图像的字节数据
    :return: 修复后的PIL图像对象
    """
    try:
        # 在色条模式下，detector会返回一个绿色覆盖的掩码图像
        masked_bytes = detector(image_bytes, is_mosaic=False)
        
        # 将掩码图像转换为PIL图像
        masked_image = Image.open(io.BytesIO(masked_bytes))
        
        # 在色条模式下，decensor期望两个参数都是相同的掩码图像
        result = decensor(masked_image, masked_image, is_mosaic=False)
        return result
    except Exception as e:
        print(f"色条自动修复模式处理失败: {e}")
        raise


def process_mosaic_auto(image_bytes: bytes) -> Image.Image:
    """
    处理马赛克自动修复模式（模式2）
    :param image_bytes: 输入图像的字节数据
    :return: 修复后的PIL图像对象
    """
    try:
        # 保存原始图像
        original_image = Image.open(io.BytesIO(image_bytes))
        
        # 获取掩码图像（字节数据）
        masked_bytes = detector(image_bytes, is_mosaic=True)
        masked_image = Image.open(io.BytesIO(masked_bytes))
        
        # 调用修复函数
        result = decensor(original_image, masked_image, is_mosaic=True)
        return result
    except Exception as e:
        print(f"马赛克自动修复模式处理失败: {e}")
        raise


def process_mosaic_esrgan_auto(image_bytes: bytes) -> Image.Image:
    """
    处理马赛克自动修复并放大模式（模式3）
    :param image_bytes: 输入图像的字节数据
    :return: 修复后的PIL图像对象
    """
    try:
        # 调用ESRGAN处理，返回两个字节数据
        org_bytes, masked_bytes = esrgan(image_bytes)
        
        # 转换为PIL图像
        original_image = Image.open(io.BytesIO(org_bytes))
        masked_image = Image.open(io.BytesIO(masked_bytes))
        
        # 调用修复函数
        result = decensor(original_image, masked_image, is_mosaic=True)
        return result
    except Exception as e:
        print(f"马赛克自动修复并放大模式处理失败: {e}")
        raise


def process_image_stream(image_bytes: bytes, mode: int) -> Image.Image:
    """
    作为组件被调用时的处理函数
    :param image_bytes: 输入图像的字节数据
    :param mode: 处理模式 (1, 2, 3)
    :return: 修复后的PIL图像对象
    """
    if mode == 1:
        return process_bar_auto(image_bytes)
    elif mode == 2:
        return process_mosaic_auto(image_bytes)
    elif mode == 3:
        return process_mosaic_esrgan_auto(image_bytes)
    else:
        raise ValueError(f"不支持的修复模式: {mode}")


def save_image_to_bytes(image: Image.Image) -> bytes:
    """
    将PIL图像转换为字节数据
    :param image: PIL图像对象
    :return: 图像的字节数据
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def get_supported_formats():
    """返回支持的图片格式"""
    return ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

def main():
    """
    命令行接口主函数
    """
    parser = argparse.ArgumentParser(description="DeepCreampy 图像修复工具（命令行版）")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出图像文件路径")
    parser.add_argument(
        "--mode", "-m", 
        required=True, 
        choices=["1", "2", "3"], 
        help="处理模式: 1=色条自动修复, 2=马赛克自动修复, 3=马赛克自动修复并放大"
    )
    
    args = parser.parse_args()
    
    # 读取输入图像
    try:
        with open(args.input, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"错误：无法读取输入文件 {args.input} - {e}")
        sys.exit(1)
    
    # 根据模式处理图像
    try:
        mode_int = int(args.mode)
        result_image = process_image_stream(image_bytes, mode_int)
    except Exception as e:
        print(f"错误：处理图像时发生异常 - {e}")
        sys.exit(1)
    
    # 保存输出图像
    try:
        result_image.save(args.output, format="PNG")
        print(f"处理完成！结果已保存到 {args.output}")
    except Exception as e:
        print(f"错误：无法保存输出文件 {args.output} - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()