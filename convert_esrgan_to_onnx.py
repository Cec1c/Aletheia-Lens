"""
ESRGAN PyTorch 模型 → ONNX 格式转换脚本
用法: python convert_esrgan_to_onnx.py
输出: models/esrgan/4x-Fatal-Pixels.onnx
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from ColabESRGAN.architecture import RRDB_Net


def convert_esrgan_to_onnx():
    pth_path = "models/esrgan/4x-Fatal-Pixels.pth"
    onnx_path = "models/esrgan/4x-Fatal-Pixels.onnx"

    if not os.path.exists(pth_path):
        print(f"ERROR: 模型文件不存在: {pth_path}")
        sys.exit(1)

    print(f"[1/4] 加载 PyTorch 模型: {pth_path}")
    model = RRDB_Net(
        in_nc=3, out_nc=3, nf=64, nb=23, gc=32, upscale=4,
        norm_type=None, act_type='leakyrelu', mode='CNA',
        res_scale=1, upsample_mode='upconv'
    )
    model.load_state_dict(torch.load(pth_path, map_location='cpu'), strict=True)
    model.eval()
    print(f"      模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("[2/4] 创建测试输入并 trace...")
    # ESRGAN 输入: 缩小后的小图，尺寸不定，用 64x64 做 dummy
    dummy_input = torch.randn(1, 3, 64, 64)

    # 验证 forward 能正常跑
    with torch.no_grad():
        output = model(dummy_input)
    print(f"      测试通过: 输入 {dummy_input.shape} → 输出 {output.shape}")

    print("[3/4] 导出 ONNX (opset=17)...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'},
        },
    )
    print(f"      导出完成: {onnx_path}")

    print("[4/4] 验证 ONNX 模型...")
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("      ONNX 模型验证通过 ✓")

    # 测试 ONNX Runtime 推理一致性
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {'input': dummy_input.numpy()}
    ort_output = session.run(['output'], ort_inputs)[0]

    torch_output = output.detach().cpu().numpy()
    if not np.isfinite(torch_output).all() or not np.isfinite(ort_output).all():
        raise RuntimeError("ESRGAN conversion produced non-finite output")
    if float(np.ptp(ort_output)) <= 1e-8:
        raise RuntimeError("ESRGAN conversion produced degenerate constant output")

    diff = np.abs(torch_output - ort_output).max()
    print(f"      PyTorch vs ONNX 最大误差: {diff:.8f}")
    if diff >= 1e-4:
        raise RuntimeError(
            f"ESRGAN conversion accuracy check failed: max error {diff:.2e}"
        )
    print("      精度一致 ✓")

    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n完成! ONNX 模型: {onnx_path} ({onnx_size_mb:.1f} MB)")


if __name__ == "__main__":
    convert_esrgan_to_onnx()
