# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 构建配置文件 (ONNX Runtime 统一推理版本)
# 构建命令: pyinstaller main.spec
# 体积: ~500MB (不含模型文件约 250MB + 模型 ~250MB)

import os
import shutil
import site
import sys
from pathlib import Path

# 收集模型文件 (仅 ONNX 格式)
def get_model_datas():
    """递归收集 models 目录下的 ONNX 模型文件"""
    datas = []
    models_dir = Path('models')
    if models_dir.exists():
        for pattern in ('*.onnx', '*.onnx.data'):
            for f in models_dir.rglob(pattern):
                dest_dir = str(f.parent)
                datas.append((str(f), dest_dir))

    # 字体文件
    if Path('font/sckkt.ttf').exists():
        datas.append(('font/sckkt.ttf', 'font'))
    if Path('ico.ico').exists():
        datas.append(('ico.ico', '.'))

    return datas


def get_gpu_binaries():
    """Bundle CUDA/cuDNN wheels only for the dedicated cuda12 artifact."""
    if os.environ.get('ALETHEIA_BUILD_FLAVOR') != 'cuda12':
        return []

    binaries = []
    roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(user_site)

    seen = set()
    for root in roots:
        nvidia_dir = Path(root) / 'nvidia'
        if not nvidia_dir.exists():
            continue
        for dll in nvidia_dir.rglob('*.dll'):
            resolved = str(dll.resolve())
            if resolved not in seen:
                seen.add(resolved)
                binaries.append((resolved, '.'))
    return binaries


def remove_nested_nvidia_duplicates(binaries):
    """Keep the complete root DLL set and drop hook-generated second copies."""
    if os.environ.get('ALETHEIA_BUILD_FLAVOR') != 'cuda12':
        return binaries

    return [
        binary
        for binary in binaries
        if str(binary[0]).replace('\\', '/').split('/', 1)[0].casefold() != 'nvidia'
    ]


def get_archive_binaries():
    """Bundle 7-Zip so packaged RAR extraction does not depend on the host."""
    program_files = Path(os.environ.get('ProgramFiles', r'C:\Program Files'))
    candidates = [program_files / '7-Zip' / '7z.exe']
    discovered = shutil.which('7z.exe')
    if discovered:
        candidates.append(Path(discovered))

    seven_zip_exe = next((path for path in candidates if path.exists()), None)
    if seven_zip_exe is None:
        raise FileNotFoundError(
            "7-Zip is required to build RAR support; install 7-Zip or add 7z.exe to PATH"
        )

    seven_zip_dir = seven_zip_exe.parent
    required_files = [seven_zip_dir / filename for filename in ('7z.exe', '7z.dll')]
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"7-Zip installation is incomplete; missing: {', '.join(missing_files)}"
        )
    return [(str(source), 'tools/7zip') for source in required_files]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=get_gpu_binaries() + get_archive_binaries(),
    datas=get_model_datas(),
    hiddenimports=[
        # GUI
        'dearpygui',
        'dearpygui._dearpygui',
        # 图像处理
        'PIL',
        'PIL.Image',
        'PIL._imaging',
        'cv2',
        'numpy',
        'numpy.core',
        'numpy.core.multiarray',
        'numpy.core._multiarray_umath',
        'numpy.core._multiarray_tests',
        # ONNX Runtime (ESRGAN + deepcreampy + MaskRCNN 全部 ONNX)
        'onnxruntime',
        'onnxruntime.capi',
        'onnx_runtime',
        # mrcnn.config (纯 Python 配置类，detector.py 需要)
        'mrcnn',
        'mrcnn.config',
        # 项目模块
        'processer',
        'detector',
        'decensor',
        'esrgan',
        'predict',
        'config',
        # 工具模块
        'tools',
        'tools.image_tool',
        'tools.decorators',
        'tools.green_mask_project_mosaic_resolution',
        # deepcreampy
        'deepcreampy',
        'deepcreampy.utils',
        # 压缩包支持
        'py7zr',
        'rarfile',
        # 标准库模块
        'tkinter',
        'multiprocessing',
        'multiprocessing.pool',
        'threading',
        'io',
        'argparse',
        'zipfile',
        'tempfile',
        'shutil',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 测试/文档
        'tkinter.test',
        'numpy.testing',
        'IPython',
        'jupyter',
        'notebook',
        # 已移除的框架
        'tensorflow',
        'tensorflow.python',
        'keras',
        'h5py',
        'imgaug',
        'torch',
        'torchvision',
        'torchaudio',
        # 不需要的库
        'matplotlib',
        'scipy',
        'skimage',
        'onnx',  # onnxruntime 自带所需功能
        # 不再需要的模块
        'ColabESRGAN',
        'mrcnn.model',
        'mrcnn.utils',
        'mrcnn.visualize',
        'mrcnn.parallel_model',
        'convert_esrgan_to_onnx',
        'convert_mrcnn_to_onnx',
    ],
    noarchive=False,
    optimize=0,
)
a.binaries = remove_nested_nvidia_duplicates(a.binaries)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Aletheia-Lens',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['ico.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Aletheia-Lens',
)
