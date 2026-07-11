# GitHub Actions 发布配置

工作流会在每个 `v*` tag 上自动构建并发布两个包：

- `*_cpu.7z`：精简 CPU 版。
- `*_cuda12.7z`：内置 CUDA 12 / cuDNN 9 运行库的 NVIDIA GPU 版。

## 一次性准备模型 Release

大型 ONNX 模型不提交到 Git。先在仓库根目录执行：

```powershell
.\tools\publish_model_release.ps1
```

脚本会先按 `.github/model-assets.sha256` 校验三个文件，再创建非 latest 的
`models-v1` Release。Release 已存在时，需显式加 `-Replace` 才会覆盖资产。

工作流默认读取 `models-v1`，下载后会再次校验 SHA-256。若使用其他 tag，在仓库
Actions Variables 中设置 `MODELS_RELEASE_TAG`。

## 发布

```powershell
git tag v1.1.0
git push origin v1.1.0
```

手动运行 Workflow 时，`release_tag` 留空只生成临时 Artifacts；填写如 `v1.1.0`
则会为当前提交创建同名 Release。

## CUDA 验证边界

GitHub 托管的 Windows runner 没有 NVIDIA GPU，因此 CUDA 构建会校验完整 DLL
清单、CUDA provider 可发现性和四个模型的真实推理输出，但不能证明节点实际在显卡上
执行。发布前应在带 NVIDIA GPU 的 Windows 机器或 self-hosted runner 上运行；该
检查要求 Mask R-CNN 与 ESRGAN 保持 CUDA，同时验证固定走 CPU 的 DeepCreamPy
模型仍输出有效数值：

```powershell
Aletheia-Lens.exe --runtime-smoke-test --require-cuda-provider --require-cuda
```

正式 Workflow 使用只读构建 token，仅 Release job 获得 `contents: write`；所有 Action
均固定到完整 commit SHA。
