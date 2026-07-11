import dearpygui.dearpygui as dpg
import hashlib
import ntpath
import os
import stat
import sys
import threading
import unicodedata
import zipfile
import tempfile
import shutil
from tkinter import filedialog
import tkinter as tk
import config
# py7zr 和 rarfile 仅在压缩包模式时需要，设为可选导入
try:
    import py7zr
    _HAS_7Z = True
except ImportError:
    py7zr = None
    _HAS_7Z = False
try:
    import rarfile
    _HAS_RAR = True
except ImportError:
    rarfile = None
    _HAS_RAR = False


# 初始设置为False
PROCESSER_AVAILABLE = False
process_image_stream = None  # 全局变量存储处理函数
MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 20 * 1024**3
ARCHIVE_FREE_SPACE_RESERVE_BYTES = 512 * 1024**2
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CONIN$",
    "CONOUT$",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _png_output_name(filename):
    """Return a filename whose extension matches the PNG encoder."""
    return f"{filename}.processed.png"


def _flat_png_output_name(input_dir, relative_path, filename, mode):
    """Return a stable PNG name for a flattened source path."""
    normalized_root = os.path.normcase(os.path.realpath(input_dir)).replace("\\", "/")
    normalized_dir = "" if relative_path == "." else relative_path.replace("\\", "/").strip("/")
    source_path = f"{normalized_dir}/{filename}" if normalized_dir else filename
    source_key = f"{normalized_root}\0{source_path}\0mode={mode}"
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
    return f"{digest}.processed.png"


def _source_output_namespace(source_path, display_name, mode):
    normalized_source = os.path.normcase(os.path.realpath(source_path)).replace("\\", "/")
    digest = hashlib.sha256(
        f"{normalized_source}\0mode={mode}".encode("utf-8")
    ).hexdigest()[:12]
    return f"after_{display_name}_{digest}"


def _structured_folder_output_dir(input_dir, output_base_dir, mode):
    input_folder_name = os.path.basename(os.path.normpath(input_dir)) or "folder"
    namespace = _source_output_namespace(input_dir, input_folder_name, mode)
    return os.path.join(output_base_dir, namespace)


def _archive_output_dir(archive_path, output_base_dir, mode):
    archive_name = os.path.splitext(os.path.basename(os.path.normpath(archive_path)))[0]
    namespace = _source_output_namespace(archive_path, archive_name or "archive", mode)
    return os.path.join(output_base_dir, namespace)


def _path_is_within(path, parent):
    child_path = os.path.normcase(os.path.realpath(path))
    parent_path = os.path.normcase(os.path.realpath(parent))
    try:
        return os.path.commonpath([child_path, parent_path]) == parent_path
    except ValueError:
        return False


def _normalize_archive_member_path(member_name, temp_dir):
    if not isinstance(member_name, str) or not member_name or "\0" in member_name:
        raise ValueError("压缩包包含无效文件名")

    normalized = member_name.replace("\\", "/")
    while normalized.endswith("/"):
        normalized = normalized[:-1]
    if not normalized:
        raise ValueError("压缩包包含空路径")

    drive, _ = ntpath.splitdrive(normalized)
    if drive or normalized.startswith("/") or normalized.startswith("//"):
        raise ValueError(f"压缩包包含绝对路径: {member_name}")

    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"压缩包包含不安全路径: {member_name}")

    for part in parts:
        has_win32_invalid_character = any(
            ord(character) < 32 or character in '<>:"|?*'
            for character in part
        )
        if has_win32_invalid_character or part.endswith((" ", ".")):
            raise ValueError(f"压缩包包含 Windows 不安全文件名: {member_name}")
        reserved_key = part.rstrip(" .").split(".", 1)[0].upper()
        if reserved_key in WINDOWS_RESERVED_NAMES:
            raise ValueError(f"压缩包包含 Windows 保留文件名: {member_name}")

    target_path = os.path.join(temp_dir, *parts)
    if not _path_is_within(target_path, temp_dir):
        raise ValueError(f"压缩包成员路径越过临时目录: {member_name}")

    return unicodedata.normalize("NFC", "/".join(parts)).casefold()


def _validate_archive_members(members, temp_dir):
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError(f"压缩包文件数量超过限制: {len(members)} > {MAX_ARCHIVE_MEMBERS}")

    if any(is_link for _, _, _, is_link in members):
        raise ValueError("压缩包包含符号链接、目录联接或特殊文件，已拒绝解压")

    seen_paths = set()
    for member_name, _, _, _ in members:
        normalized_key = _normalize_archive_member_path(member_name, temp_dir)
        if normalized_key in seen_paths:
            raise ValueError(f"压缩包成员路径规范化后冲突: {member_name}")
        seen_paths.add(normalized_key)

    total_size = sum(
        max(0, int(size or 0))
        for _, is_dir, size, _ in members
        if not is_dir
    )
    if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ValueError(
            "压缩包解压后总大小超过限制: "
            f"{total_size} > {MAX_ARCHIVE_UNCOMPRESSED_BYTES} bytes"
        )

    free_space = shutil.disk_usage(temp_dir).free
    if total_size + ARCHIVE_FREE_SPACE_RESERVE_BYTES > free_space:
        raise ValueError(
            "临时目录剩余空间不足以安全解压压缩包: "
            f"需要 {total_size + ARCHIVE_FREE_SPACE_RESERVE_BYTES} bytes，"
            f"可用 {free_space} bytes"
        )


def _validate_extracted_tree(temp_dir):
    extraction_root = os.path.realpath(temp_dir)
    for current_root, directories, filenames in os.walk(extraction_root, followlinks=False):
        for name in directories + filenames:
            extracted_path = os.path.join(current_root, name)
            file_attributes = getattr(os.lstat(extracted_path), "st_file_attributes", 0)
            is_reparse_point = bool(
                file_attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
            if os.path.islink(extracted_path) or is_reparse_point:
                raise ValueError("压缩包解压结果包含链接或目录联接，已拒绝处理")
            if not _path_is_within(extracted_path, extraction_root):
                raise ValueError("压缩包解压结果越过临时目录边界，已拒绝处理")


def _validate_runtime_smoke_outputs(detection, esrgan_output, bar_output, mosaic_output):
    import numpy as np

    required_detection_keys = {"rois", "class_ids", "scores", "masks"}
    if not isinstance(detection, dict) or not required_detection_keys.issubset(detection):
        raise RuntimeError("Mask R-CNN smoke output is missing required fields")

    rois = np.asarray(detection["rois"])
    class_ids = np.asarray(detection["class_ids"])
    scores = np.asarray(detection["scores"])
    masks = np.asarray(detection["masks"])
    detection_count = rois.shape[0] if rois.ndim == 2 else -1
    if (
        rois.shape != (detection_count, 4)
        or class_ids.shape != (detection_count,)
        or scores.shape != (detection_count,)
        or masks.shape != (512, 512, detection_count)
    ):
        raise RuntimeError("Mask R-CNN smoke output has invalid shapes")
    if not all(np.isfinite(array).all() for array in (rois, class_ids, scores, masks)):
        raise RuntimeError("Mask R-CNN smoke output contains non-finite values")

    mosaic_indices = np.flatnonzero(class_ids == 2)
    if mosaic_indices.size == 0:
        raise RuntimeError("Mask R-CNN smoke output missed the synthetic mosaic")
    best_mosaic = int(mosaic_indices[np.argmax(scores[mosaic_indices])])
    if float(scores[best_mosaic]) < 0.85 or not masks[:, :, best_mosaic].any():
        raise RuntimeError("Mask R-CNN synthetic mosaic detection is too weak")

    expected_roi = np.array([160, 160, 352, 352], dtype=np.float32)
    actual_roi = rois[best_mosaic].astype(np.float32)
    intersection_y1 = max(expected_roi[0], actual_roi[0])
    intersection_x1 = max(expected_roi[1], actual_roi[1])
    intersection_y2 = min(expected_roi[2], actual_roi[2])
    intersection_x2 = min(expected_roi[3], actual_roi[3])
    intersection = max(0.0, intersection_y2 - intersection_y1) * max(
        0.0, intersection_x2 - intersection_x1
    )
    expected_area = (expected_roi[2] - expected_roi[0]) * (
        expected_roi[3] - expected_roi[1]
    )
    actual_area = (actual_roi[2] - actual_roi[0]) * (
        actual_roi[3] - actual_roi[1]
    )
    union = expected_area + actual_area - intersection
    if union <= 0 or intersection / union < 0.6:
        raise RuntimeError("Mask R-CNN synthetic mosaic ROI is incorrect")

    expected_mask = np.zeros((512, 512), dtype=np.bool_)
    expected_mask[160:352, 160:352] = True
    actual_mask = masks[:, :, best_mosaic].astype(np.bool_)
    mask_intersection = np.logical_and(actual_mask, expected_mask).sum()
    mask_union = np.logical_or(actual_mask, expected_mask).sum()
    mask_iou = mask_intersection / float(mask_union) if mask_union else 0.0
    if mask_iou < 0.75:
        raise RuntimeError(
            f"Mask R-CNN synthetic mosaic mask IoU is too low: {mask_iou:.3f}"
        )

    for label, value, expected_shape in (
        ("ESRGAN", esrgan_output, (64, 64, 3)),
        ("DeepCreamPy bar", bar_output, (256, 256, 3)),
        ("DeepCreamPy mosaic", mosaic_output, (256, 256, 3)),
    ):
        output = np.asarray(value)
        if output.shape != expected_shape:
            raise RuntimeError(
                f"{label} smoke output has invalid shape: {output.shape} != {expected_shape}"
            )
        if not np.issubdtype(output.dtype, np.number) or not np.isfinite(output).all():
            raise RuntimeError(f"{label} smoke output contains invalid values")
        if float(np.ptp(output.astype(np.float64))) <= 1e-6:
            raise RuntimeError(f"{label} smoke output is degenerate")

# 关于内容字符串
ABOUT_CONTENT = """Aletheia Lens
阿勒西娅之镜
版本: 1.0-251121

我们拜请拾滩鸦，明晓失物之神
所求之物，诉说着生育的奥秘，是为杯
消去遮拦直视事物的存在，还其应有之形，
亦是打开阻拦的门扉，是为启
（https://mansus.huijiwiki.com/wiki/拾滩鸦）
    
"""

class DeepCreampyApp:
    def __init__(self):
        self.processing = False
        self.current_file_index = 0
        self.total_files = 0
        self.input_path = ""
        self.output_path = ""
        self.mode = 1
        self.input_type = "image"  # 默认改为图片模式
        self.preserve_structure = True
        self.model_loaded = False
        self.runtime_status_signature = None
        
        # 初始化Dear PyGui
        dpg.create_context()
        icon_path = config.resource_path('ico.ico')
        dpg.create_viewport(
            title='Aletheia Lens',
            small_icon=icon_path,
            large_icon=icon_path,
            width=600,
            height=780,
        )
        dpg.setup_dearpygui()
        
        # 设置字体
        self.setup_font()
        
        # 设置主题
        self.setup_theme()
        
        # 创建主窗口
        self.create_main_window()
        
        # 异步导入processer模块
        self.start_async_import()
        
    def setup_font(self):
        """设置中文字体"""
        with dpg.font_registry():
            # 尝试加载指定字体文件
            font_paths = [
                config.resource_path("font/sckkt.ttf"),
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "/System/Library/Fonts/PingFang.ttc",  # macOS 苹方
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
            ]
            
            font_loaded = False
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        with dpg.font(font_path, 18) as font1:
                            dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                            dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Simplified_Common)
                            dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
                        dpg.bind_font(font1)
                        print(f"成功加载字体: {font_path}")
                        font_loaded = True
                        break
                    except Exception as e:
                        print(f"加载字体失败 {font_path}: {e}")
                else:
                    print(f"字体文件未找到: {font_path}")
            
            if not font_loaded:
                print("使用默认字体")
    
    def setup_theme(self):
        """设置自定义主题"""
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (60, 60, 70, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (80, 80, 100, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (100, 100, 120, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (120, 120, 140, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Header, (80, 80, 100, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (100, 100, 120, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (120, 120, 140, 255), category=dpg.mvThemeCat_Core)
        
        dpg.bind_theme(global_theme)
    
    def start_async_import(self):
        """异步导入processer模块"""
        def import_processer():
            global PROCESSER_AVAILABLE, process_image_stream
            try:
                # 导入processer模块
                from processer import process_image_stream as imported_process_image_stream
                process_image_stream = imported_process_image_stream
                PROCESSER_AVAILABLE = True
                self.model_loaded = True

                self.log_runtime_status(force=True)
                
                # 更新UI状态
                self.update_processer_status()
                self.log_message("processer模块加载成功，模型预热完成")
                
            except ImportError as e:
                PROCESSER_AVAILABLE = False
                process_image_stream = None
                self.log_message(f"警告：无法导入processer模块: {e}")
                self.update_processer_status()
            except Exception as e:
                PROCESSER_AVAILABLE = False
                process_image_stream = None
                self.log_message(f"模型加载过程中发生错误: {e}")
                self.update_processer_status()
        
        # 在新线程中导入
        thread = threading.Thread(target=import_processer)
        thread.daemon = True
        thread.start()

    def log_runtime_status(self, force=False):
        """Log provider changes, including execution-time CUDA fallback."""
        from onnx_runtime import get_runtime_status

        runtime_status = get_runtime_status()
        session_providers = runtime_status.get("session_providers", {})
        signature = (
            runtime_status["device"],
            tuple(
                (model_path, tuple(providers))
                for model_path, providers in sorted(session_providers.items())
            ),
        )
        if not force and signature == getattr(self, "runtime_status_signature", None):
            return
        self.runtime_status_signature = signature

        if runtime_status["device"] == "CUDA":
            self.log_message("ONNX Runtime 已启用 CUDA GPU 加速")
        elif runtime_status["device"] == "MIXED":
            intentional_cpu_models = runtime_status.get("intentional_cpu_models", [])
            unexpected_cpu_models = runtime_status.get("unexpected_cpu_models")
            if unexpected_cpu_models is None:
                unexpected_cpu_models = [
                    model_path
                    for model_path, providers in session_providers.items()
                    if "CUDAExecutionProvider" not in providers
                    and model_path not in intentional_cpu_models
                ]
            status_parts = []
            if intentional_cpu_models:
                status_parts.append(
                    "兼容性 CPU: "
                    + ", ".join(os.path.basename(path) for path in intentional_cpu_models)
                )
            if unexpected_cpu_models:
                status_parts.append(
                    "CPU 回退: "
                    + ", ".join(os.path.basename(path) for path in unexpected_cpu_models)
                )
            suffix = f"，{'；'.join(status_parts)}" if status_parts else ""
            self.log_message(f"ONNX Runtime 部分模型使用 CUDA{suffix}")
        else:
            self.log_message("ONNX Runtime 当前使用 CPU")

        if runtime_status["guidance"]:
            self.log_message(runtime_status["guidance"])
    
    def update_processer_status(self):
        """更新processer模块状态显示"""
        if PROCESSER_AVAILABLE:
            # 更新状态为可用
            dpg.set_value("processer状态", "可用")
            dpg.configure_item("processer状态", color=(0, 255, 0, 255))
            
            # 检查模型文件是否存在
            self.check_model_files()
            
            # 隐藏下载按钮
            dpg.configure_item("processer状态_按钮", show=False)
        else:
            # 保持原来的状态
            dpg.set_value("processer状态", "加载失败")
            dpg.configure_item("processer状态", color=(255, 0, 0, 255))  # 红色
    
    def check_model_files(self):
        """检查模型文件是否存在"""
        # 检查放大模型
        esrgan_model_files = [config.esrgan_model, config.esrgan_model_data]
        if all(os.path.exists(path) for path in esrgan_model_files):
            dpg.set_value("放大模型状态", "可用")
            dpg.configure_item("放大模型状态", color=(0, 255, 0, 255))
        else:
            dpg.set_value("放大模型状态", "不可用")
            dpg.configure_item("放大模型状态", color=(255, 0, 0, 255))
        
        # 检查检测模型
        if os.path.exists(config.mrcnn_model):
            dpg.set_value("检测模型状态", "可用")
            dpg.configure_item("检测模型状态", color=(0, 255, 0, 255))
        else:
            dpg.set_value("检测模型状态", "不可用")
            dpg.configure_item("检测模型状态", color=(255, 0, 0, 255))
    
    def show_about_dialog(self):
        """显示关于对话框"""
        with dpg.window(label="神秘彩蛋？", modal=True, tag="关于窗口", width=600, height=280,no_resize=True,autosize=True,pos=(100,200)):
            dpg.add_text(ABOUT_CONTENT)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="关闭", callback=lambda: dpg.delete_item("关于窗口"))
    
    def create_main_window(self):
        """创建主窗口和所有控件"""
        with dpg.window(tag="主窗口", label="Aletheia Lens",autosize=True):
            
            # 1. 文件选择部分 - 合并输入输出
            with dpg.collapsing_header(label="在这里选择要处理的素材或图片路径", default_open=True):
                # 输入类型选择
                with dpg.group(horizontal=True):
                    dpg.add_text("输入类型:")
                    dpg.add_radio_button(
                        items=["单图片模式", "文件夹模式", "压缩包模式"],
                        tag="输入类型选择",
                        default_value="单图片模式",  # 默认图片模式
                        callback=self.on_input_type_change
                    )
                    
                dpg.add_text("选择文件夹模式会自动遍历文件夹下所有的图片文件\n同时不会改变文件夹结构")
                # 输入路径
                with dpg.group(horizontal=True):
                    dpg.add_text("输入路径:")
                    dpg.add_input_text(
                        tag="输入路径",
                        hint="选择输入路径...",
                        width=400
                    )
                    dpg.add_button(
                        label="选择",
                        callback=self.browse_input
                    )
                
                # 输出路径
                with dpg.group(horizontal=True):
                    dpg.add_text("输出路径:")
                    dpg.add_input_text(
                        tag="输出路径",
                        hint="选择输出路径...",
                        width=400
                    )
                    dpg.add_button(
                        label="选择",
                        callback=self.browse_output
                    )
            
            # 2. 处理模式选择
            with dpg.collapsing_header(label="在这里选择处理模式", default_open=True):
                dpg.add_text("一般情况下如大部分打黑条的本子请选模式I")
                dpg.add_text("对于马赛克，越厚码越奇怪和诡异，模式III慎用")
                dpg.add_radio_button(
                    items=[
                        "模式I: 色条自动修复",
                        "模式II: 马赛克自动修复", 
                        "模式III: 马赛克修复并放大"
                    ],
                    tag="模式选择",
                    default_value="模式I: 色条自动修复",
                    callback=self.on_mode_change
                )
                dpg.add_text("模式III会毁坏透明背景，游戏素材等含透明图层的慎用")
            
            # 3. 文件夹选项（仅文件夹模式显示）
            with dpg.collapsing_header(label="在这里调整文件夹选项", tag="文件夹选项区域", show=False):
                dpg.add_checkbox(
                    label="保留文件夹结构（在顶层文件夹前添加'after_'前缀）",
                    tag="保留结构复选框",
                    default_value=True,
                    callback=self.on_preserve_structure_change
                )
                dpg.add_text("启用时：创建'after_原文件夹名'并保持相同目录结构\n禁用时：所有处理后的图片直接放在输出文件夹中", 
                           color=(150, 150, 150, 255))
            
            # 4. 统计信息（仅文件夹模式显示）
            with dpg.collapsing_header(label="现在处理得怎么样啦？", tag="统计信息区域", show=False):
                with dpg.group(horizontal=True):
                    dpg.add_text("总共我要处理这么多:")
                    dpg.add_text("0", tag="图片数量")
                
                with dpg.group(horizontal=True):
                    dpg.add_text("现在我处理的这么多:")
                    dpg.add_text("0/0", tag="进度文本")
                
                dpg.add_progress_bar(
                    tag="进度条",
                    default_value=0.0,
                    width=-1
                )
            
            # 5. 模块状态检测
            with dpg.collapsing_header(label="这些东西必须装好才能用哈", default_open=True):
                dpg.add_text("模块可用性:")
                
                # processer模块状态
                with dpg.group(horizontal=True):
                    dpg.add_text("主要模块(processer):")
                    dpg.add_text("加载中...", tag="processer状态", color=(255, 165, 0, 255))
                    dpg.add_button(
                        label="重新加载",
                        tag="processer状态_按钮",
                        callback=self.reload_processer,
                        user_data="processer模块"
                    )
                
                # 模型状态
                models = [
                    ("放大模型(4x-Fatal-Pixels)", "放大模型状态"),
                    ("检测模型(weights)", "检测模型状态")
                ]
                
                for model_name, tag in models:
                    with dpg.group(horizontal=True):
                        dpg.add_text(f"{model_name}:")
                        dpg.add_text("等待processer...", tag=tag, color=(150, 150, 150, 255))
                        dpg.add_button(
                            label="下载",
                            tag=f"{tag}_按钮",
                            callback=lambda s, a, u: self.download_model(u),
                            user_data=model_name
                        )
            
            # 6. 执行按钮和关于按钮
            dpg.add_spacer(height=15)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=150)  # 居中对齐
                dpg.add_button(
                    label="点我开始一键去码",
                    tag="执行按钮",
                    callback=self.execute_processing,
                    width=200,  # 缩小宽度
                    height=35
                )
                dpg.add_spacer(width=150)
                dpg.add_spacer(height=20)  # 按钮间距
                dpg.add_button(
                    label="O.O",
                    tag="关于按钮",
                    callback=self.show_about_dialog,
                    width=30,  # 较小的关于按钮
                    height=25
                )
            
            # 7. 日志输出
            with dpg.collapsing_header(label="告诉你我做了什么", default_open=True):
                dpg.add_input_text(
                    tag="日志输出",
                    multiline=True,
                    readonly=True,
                    height=120,
                    width=-1
                )
    
    def on_input_type_change(self, sender, app_data):
        """输入类型改变回调"""
        input_type_map = {
            "单图片模式": "image",
            "文件夹模式": "folder",
            "压缩包模式": "archive"
        }
        self.input_type = input_type_map.get(app_data, "image")
        
        # 显示/隐藏文件夹选项和统计信息
        if self.input_type == "folder":
            dpg.configure_item("文件夹选项区域", show=True)
            dpg.configure_item("统计信息区域", show=True)
            self.update_image_count()
        else:
            dpg.configure_item("文件夹选项区域", show=False)
            dpg.configure_item("统计信息区域", show=False)
    
    def on_mode_change(self, sender, app_data):
        """模式选择回调"""
        mode_map = {
            "模式I: 色条自动修复": 1,
            "模式II: 马赛克自动修复": 2,
            "模式III: 马赛克修复并放大": 3
        }
        self.mode = mode_map.get(app_data, 1)
    
    def on_preserve_structure_change(self, sender, app_data):
        """保留结构复选框回调"""
        self.preserve_structure = app_data
    
    def browse_input(self):
        """浏览输入路径"""
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        if self.input_type == "folder":
            path = filedialog.askdirectory(title="选择输入文件夹")
        elif self.input_type == "archive":
            path = filedialog.askopenfilename(
                title="选择压缩包",
                filetypes=[("压缩包文件", "*.zip;*.7z;*.rar")]
            )
        else:
            path = filedialog.askopenfilename(
                title="选择输入图片",
                filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
            )
        
        if path:
            dpg.set_value("输入路径", path)
            self.input_path = path
            if self.input_type == "folder":
                self.update_image_count()
    
    def browse_output(self):
        """浏览输出路径"""
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            dpg.set_value("输出路径", path)
            self.output_path = path
    
    def update_image_count(self):
        """更新图片数量统计"""
        if not self.input_path or not os.path.isdir(self.input_path):
            dpg.set_value("图片数量", "0")
            return
        
        # 递归统计所有图片文件
        count = self.count_images_recursive(self.input_path)
        dpg.set_value("图片数量", str(count))
        self.total_files = count
    
    def count_images_recursive(self, folder_path):
        """递归统计文件夹中的所有图片文件"""
        count = 0
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    count += 1
        return count
    
    def download_model(self, model_name):
        """下载模型（占位函数）"""
        self.log_message("开始自动下载模型...骗你的，这个功能我还没做")

    
    def reload_processer(self):
        """重新加载processer模块"""
        self.log_message("正在重新加载processer模块...")
        self.start_async_import()
    
    def execute_processing(self):
        """执行处理"""
        if self.processing:
            return
        
        # 验证输入
        self.input_path = dpg.get_value("输入路径")
        self.output_path = dpg.get_value("输出路径")
        
        if not self.input_path:
            self.log_message("错误：请选择输入路径")
            return
        
        if not self.output_path:
            self.log_message("错误：请选择输出路径")
            return
        
        if not PROCESSER_AVAILABLE or process_image_stream is None:
            self.log_message("错误：processer模块不可用，无法进行处理")
            return
        
        # 禁用执行按钮
        dpg.configure_item("执行按钮", enabled=False)
        self.processing = True
        
        # 在新线程中执行处理
        thread = threading.Thread(target=self.process_files)
        thread.daemon = True
        thread.start()
    
    def process_files(self):
        """处理文件的主要逻辑"""
        try:
            if self.input_type == "folder":
                self.process_folder()
            elif self.input_type == "archive":
                self.process_archive()
            else:
                self.process_single_file()
                
            self.log_message("处理完成！")
            
        except Exception as e:
            self.log_message(f"处理过程中发生错误: {str(e)}")
        finally:
            # 恢复UI状态
            self.processing = False
            dpg.configure_item("执行按钮", enabled=True)
    
    def process_folder(self):
        """处理文件夹中的所有图片，可选择是否保留结构"""
        input_dir = self.input_path
        output_base_dir = self.output_path

        effective_output_dir = (
            _structured_folder_output_dir(input_dir, output_base_dir, self.mode)
            if self.preserve_structure
            else output_base_dir
        )
        if _path_is_within(effective_output_dir, input_dir):
            self.log_message("错误：输出目录不能位于输入目录内，请选择输入目录之外的位置")
            return
        
        image_files = self.get_all_image_files(input_dir)
        self.total_files = len(image_files)
        self.current_file_index = 0
        
        if self.total_files == 0:
            self.log_message("错误：在输入文件夹中未找到任何图片文件")
            return
        
        if self.preserve_structure:
            # 保留文件夹结构模式
            self.process_with_structure(input_dir, output_base_dir, image_files)
        else:
            # 平铺模式（所有文件放在输出文件夹根目录）
            self.process_flat(input_dir, output_base_dir, image_files)
    
    def get_all_image_files(self, folder_path):
        """递归获取文件夹中的所有图片文件"""
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    relative_path = os.path.relpath(root, folder_path)
                    image_files.append((root, file, relative_path))
        return image_files
    
    def extract_archive(self, archive_path):
        """解压压缩包到临时目录"""
        temp_dir = tempfile.mkdtemp()
        os.makedirs(temp_dir, exist_ok=True)
        archive_ext = os.path.splitext(archive_path)[1].lower()

        try:
            if archive_ext == ".zip":
                with zipfile.ZipFile(archive_path, "r") as archive:
                    members = archive.infolist()
                    _validate_archive_members(
                        [
                            (
                                member.filename,
                                member.is_dir(),
                                member.file_size,
                                stat.S_ISLNK(member.external_attr >> 16),
                            )
                            for member in members
                        ],
                        temp_dir,
                    )
                    archive.extractall(temp_dir)
            elif archive_ext == ".7z":
                if not _HAS_7Z:
                    raise ImportError("需要安装 py7zr 库: pip install py7zr")
                with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                    members = archive.files
                    _validate_archive_members(
                        [
                            (
                                member.filename,
                                member.is_directory,
                                member.uncompressed,
                                member.is_symlink or member.is_junction or member.is_socket,
                            )
                            for member in members
                        ],
                        temp_dir,
                    )
                    archive.extractall(path=temp_dir)
            elif archive_ext == ".rar":
                if not _HAS_RAR:
                    raise ImportError("需要安装 rarfile 库: pip install rarfile")
                bundled_seven_zip = config.resource_path("tools/7zip/7z.exe")
                if os.path.exists(bundled_seven_zip):
                    rarfile.SEVENZIP_TOOL = bundled_seven_zip
                    rarfile.tool_setup(force=True)
                with rarfile.RarFile(archive_path, "r") as archive:
                    members = archive.infolist()
                    _validate_archive_members(
                        [
                            (
                                member.filename,
                                member.is_dir(),
                                member.file_size,
                                member.is_symlink() or bool(getattr(member, "file_redir", None)),
                            )
                            for member in members
                        ],
                        temp_dir,
                    )
                    archive.extractall(temp_dir)
            else:
                raise ValueError("不支持的压缩包格式")
            _validate_extracted_tree(temp_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        return temp_dir

    def process_archive(self):
        """处理压缩包中的所有图片并保留内部目录结构"""
        temp_dir = None

        try:
            self.log_message("开始解压压缩包")
            temp_dir = self.extract_archive(self.input_path)
            self.log_message(f"压缩包已解压到临时目录: {temp_dir}")

            image_files = self.get_all_image_files(temp_dir)
            self.total_files = len(image_files)
            self.current_file_index = 0

            if self.total_files == 0:
                self.log_message("错误：压缩包中未找到任何图片文件")
                return

            output_dir = _archive_output_dir(self.input_path, self.output_path, self.mode)
            os.makedirs(output_dir, exist_ok=True)
            self.log_message("开始处理压缩包，保留内部目录结构")
            self.log_message(f"输出目录: {output_dir}")

            for i, (root, filename, relative_path) in enumerate(image_files):
                if not self.processing:
                    self.log_message("处理已停止")
                    break

                input_path = os.path.join(root, filename)
                output_relative_dir = os.path.join(output_dir, relative_path)
                output_path = os.path.join(output_relative_dir, _png_output_name(filename))

                os.makedirs(output_relative_dir, exist_ok=True)

                try:
                    with open(input_path, "rb") as f:
                        image_bytes = f.read()

                    # 使用全局的process_image_stream函数处理图片
                    result_image = process_image_stream(image_bytes, self.mode)
                    self.log_runtime_status()
                    result_image.save(output_path, format="PNG")

                    self.log_message(f"已处理: {os.path.join(relative_path, filename)}")

                except Exception as e:
                    self.log_message(f"处理文件 {os.path.join(relative_path, filename)} 时出错: {e}")
                    continue

                # 更新进度
                self.current_file_index = i + 1
                progress = (self.current_file_index / self.total_files) * 100
                self.update_progress(progress, self.current_file_index, self.total_files)
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
                self.log_message("已清理临时解压目录")

    def process_with_structure(self, input_dir, output_base_dir, image_files):
        """处理文件夹并保留结构"""
        output_dir = _structured_folder_output_dir(input_dir, output_base_dir, self.mode)
        
        self.log_message(f"开始处理文件夹，保留结构模式")
        self.log_message(f"输出目录: {output_dir}")
        
        # 处理每个文件
        for i, (root, filename, relative_path) in enumerate(image_files):
            if not self.processing:
                self.log_message("处理已停止")
                break
                
            # 构建输入和输出路径
            input_path = os.path.join(root, filename)
            
            # 保持相对路径结构
            output_relative_dir = os.path.join(output_dir, relative_path)
            output_path = os.path.join(output_relative_dir, _png_output_name(filename))
            
            # 确保输出目录存在
            os.makedirs(output_relative_dir, exist_ok=True)
            
            try:
                # 处理图片
                with open(input_path, "rb") as f:
                    image_bytes = f.read()
                
                # 使用全局的process_image_stream函数处理图片
                result_image = process_image_stream(image_bytes, self.mode)
                self.log_runtime_status()
                result_image.save(output_path, format="PNG")
                
                self.log_message(f"已处理: {os.path.join(relative_path, filename)}")
                
            except Exception as e:
                self.log_message(f"处理文件 {os.path.join(relative_path, filename)} 时出错: {e}")
                continue
            
            # 更新进度
            self.current_file_index = i + 1
            progress = (self.current_file_index / self.total_files) * 100
            self.update_progress(progress, self.current_file_index, self.total_files)
    
    def process_flat(self, input_dir, output_dir, image_files):
        """处理文件夹但不保留结构（平铺模式）"""
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.log_message(f"开始处理文件夹，平铺模式")
        self.log_message(f"输出目录: {output_dir}")
        
        # 处理每个文件
        for i, (root, filename, relative_path) in enumerate(image_files):
            if not self.processing:
                self.log_message("处理已停止")
                break
                
            input_path = os.path.join(root, filename)
            
            output_filename = _flat_png_output_name(
                input_dir,
                relative_path,
                filename,
                self.mode,
            )
            
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                # 处理图片
                with open(input_path, "rb") as f:
                    image_bytes = f.read()
                
                # 使用全局的process_image_stream函数处理图片
                result_image = process_image_stream(image_bytes, self.mode)
                self.log_runtime_status()
                result_image.save(output_path, format="PNG")
                
                self.log_message(f"已处理: {os.path.join(relative_path, filename)} -> {output_filename}")
                
            except Exception as e:
                self.log_message(f"处理文件 {os.path.join(relative_path, filename)} 时出错: {e}")
                continue
            
            # 更新进度
            self.current_file_index = i + 1
            progress = (self.current_file_index / self.total_files) * 100
            self.update_progress(progress, self.current_file_index, self.total_files)
    
    def process_single_file(self):
        """处理单个图片文件"""
        input_path = self.input_path
        output_dir = self.output_path
        
        # 生成输出文件名
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"processed_{name}.png")
        
        try:
            with open(input_path, "rb") as f:
                image_bytes = f.read()
            
            # 使用全局的process_image_stream函数处理图片
            result_image = process_image_stream(image_bytes, self.mode)
            self.log_runtime_status()
            result_image.save(output_path, format="PNG")
            
            self.log_message(f"已处理: {filename}")
            
            # 更新进度到100%
            self.update_progress(100, 1, 1)
            
        except Exception as e:
            raise Exception(f"处理文件失败: {str(e)}")
    
    def update_progress(self, progress, current, total):
        """更新进度显示"""
        dpg.set_value("进度条", progress / 100)
        dpg.set_value("进度文本", f"{current}/{total}")
    
    def log_message(self, message):
        """添加日志消息"""
        current_log = dpg.get_value("日志输出")
        new_log = f"{message}\n{current_log}"
        dpg.set_value("日志输出", new_log)
    
    def run(self):
        """运行应用程序"""
        dpg.show_viewport()
        dpg.set_primary_window("主窗口", True)
        
        # 显示启动消息
        self.log_message("主程序已启动")
        self.log_message("正在异步加载主要处理模块，请稍候...")
        
        dpg.start_dearpygui()
        dpg.destroy_context()

def run_runtime_smoke_test(require_cuda=False, require_cuda_provider=False):
    """Run one real inference through every packaged ONNX model."""
    import numpy as np
    import detector
    import esrgan
    import predict
    from onnx_runtime import get_runtime_status

    detector_input = np.full((512, 512, 3), 255, dtype=np.uint8)
    checkerboard = (
        np.indices((12, 12)).sum(axis=0) % 2 * 128
    ).astype(np.uint8)
    mosaic_patch = np.repeat(np.repeat(checkerboard, 16, axis=0), 16, axis=1)
    detector_input[160:352, 160:352, :] = mosaic_patch[:, :, None]
    detection = detector.detect_image(detector_input)

    gradient_axis = np.linspace(0, 255, 16, dtype=np.uint8)
    gradient_x, gradient_y = np.meshgrid(gradient_axis, gradient_axis, indexing="xy")
    esrgan_input = np.stack(
        [gradient_x, gradient_y, np.full((16, 16), 128, dtype=np.uint8)],
        axis=-1,
    )
    esrgan_output = esrgan._run_esrgan_onnx(esrgan_input)
    censored = np.linspace(
        -1.0,
        1.0,
        256 * 256 * 3,
        dtype=np.float32,
    ).reshape((256, 256, 3))
    mask = np.zeros_like(censored)
    mask[64:192, 64:192, :] = 1.0
    bar_output = predict.predict(censored, mask, False)
    mosaic_output = predict.predict(censored, mask, True)
    _validate_runtime_smoke_outputs(
        detection,
        esrgan_output,
        bar_output,
        mosaic_output,
    )

    runtime_status = get_runtime_status()
    if require_cuda_provider and not runtime_status["cuda_available"]:
        raise RuntimeError(
            "CUDA smoke test required CUDAExecutionProvider to be available: "
            f"{runtime_status.get('available_providers', [])}"
        )
    if require_cuda:
        session_providers = runtime_status.get("session_providers", {})
        required_cuda_models = [config.mrcnn_model, config.esrgan_model]
        missing_cuda_models = [
            os.path.basename(model_path)
            for model_path in required_cuda_models
            if "CUDAExecutionProvider" not in session_providers.get(str(model_path), [])
        ]
    else:
        missing_cuda_models = []
    if missing_cuda_models:
        raise RuntimeError(
            "CUDA smoke test required all CUDA-compatible models on CUDA; missing: "
            f"{', '.join(missing_cuda_models)}. "
            f"Sessions: {runtime_status.get('session_providers', {})}"
        )
    if sys.stdout is not None:
        print(f"Runtime smoke test passed: {runtime_status}")
    return runtime_status


def _write_runtime_smoke_report(contents):
    report_path = os.environ.get("ALETHEIA_SMOKE_REPORT")
    if not report_path:
        return
    report_dir = os.path.dirname(os.path.abspath(report_path))
    os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report:
        report.write(contents)


def main(argv=None):
    """主函数"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--runtime-smoke-test" in argv:
        try:
            runtime_status = run_runtime_smoke_test(
                require_cuda="--require-cuda" in argv,
                require_cuda_provider="--require-cuda-provider" in argv,
            )
        except Exception as exc:
            import traceback

            report = f"FAIL: {exc}\n{traceback.format_exc()}"
            _write_runtime_smoke_report(report)
            if sys.stderr is not None:
                print(f"Runtime smoke test failed: {exc}", file=sys.stderr)
            return 1
        _write_runtime_smoke_report(f"PASS\n{runtime_status!r}\n")
        return 0

    app = DeepCreampyApp()
    app.run()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
