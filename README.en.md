<div align="center">

<img src="readmeimg/logo.png" alt="Aletheia Lens purple lens icon" width="128" height="128">

# Aletheia Lens

[简体中文](README.md) ｜ [English](README.en.md)

[![Release](https://img.shields.io/github/v/release/Cec1c/Aletheia-Lens?style=flat-square)](https://github.com/Cec1c/Aletheia-Lens/releases/latest) [![License](https://img.shields.io/github/license/Cec1c/Aletheia-Lens?style=flat-square&color=blue)](LICENSE) [![Downloads](https://img.shields.io/github/downloads/Cec1c/Aletheia-Lens/total?style=flat-square)](https://github.com/Cec1c/Aletheia-Lens/releases) [![Stars](https://img.shields.io/github/stars/Cec1c/Aletheia-Lens?style=flat-square)](https://github.com/Cec1c/Aletheia-Lens/stargazers)

A local image restoration tool powered by DeepCreamPy, hent-AI, and ONNX Runtime.

Detect covered regions and process individual images, folders, or archives with a Windows desktop interface and ready-to-run release packages.

[Download](https://github.com/Cec1c/Aletheia-Lens/releases/latest) ｜ [Quick start](#quick-start) ｜ [Usage](#usage) ｜ [Experimental features](#experimental-features) ｜ [Report an issue](#report-an-issue)

</div>

> [!NOTE]
> Designed for anime-style illustrations only. This README describes the current source code; check the release notes for features available in a downloaded version. The desktop interface is currently in Chinese; this guide includes the relevant button labels.

## What you can do

- **Restore a batch of images with censorship bars**: automatically detect and repair covered regions, reducing repetitive work.
- **Process extracted game assets**: traverse nested folders or archives while preserving their structure in a separate output directory.
- **Process scanned manga**: optionally remove screentones before restoration.

## Preview

<table>
  <tr>
    <td align="center" valign="top">
      <strong>Main window and experimental features</strong><br>
      <img src="readmeimg/main-window.png" alt="Aletheia Lens main window with input paths, restoration modes, and the experimental feature warning" width="380">
    </td>
    <td align="center" valign="top">
      <strong>Manual mask editor (experimental)</strong><br>
      <img src="readmeimg/manual-editor.png" alt="Synthetic test image demonstrating brush selections, zoom, undo, and manual restoration" width="480"><br>
      <strong>Result from the selected mask</strong><br>
      <img src="readmeimg/manual-result.png" alt="Only the selected upper bar is repaired; the unselected lower bar and original green region remain unchanged" width="480">
    </td>
  </tr>
</table>

## Quick start

### Download and run

Choose a package from [Releases](https://github.com/Cec1c/Aletheia-Lens/releases/latest):

| Package | Best suited for | Details |
| --- | --- | --- |
| `*_cpu.7z` | Devices without a compatible NVIDIA GPU, or users who prefer a smaller download | Runs inference on the CPU |
| `*_cuda12.7z` | NVIDIA GPUs compatible with CUDA 12 | Includes CUDA 12 / cuDNN 9 libraries; detection and upscaling prefer the GPU |

1. Extract the archive and run `Aletheia-Lens.exe`.
2. Wait for the processing modules and models to load.
3. Choose the input type, input path, output folder, and restoration mode, then click “点我开始一键去码” (start processing).

Release packages include the required models. You do not need to install Python or download models separately.

> [!IMPORTANT]
> DeepCreamPy's `bar.onnx` and `mosaic.onnx` remain pinned to the CPU in the CUDA package: these older models produce non-finite values with CUDA12/cuDNN9. Logs show whether ONNX sessions use CUDA, a mix of CPU and CUDA, or CPU only, and explain this compatibility policy.

### Run from source

The project is built with **Python 3.10.11**. Compatibility with other Python versions has not been confirmed.

```powershell
git clone https://github.com/Cec1c/Aletheia-Lens.git
cd Aletheia-Lens
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For an NVIDIA GPU, replace the last command with:

```powershell
pip install -r requirements-gpu.txt
```

To process RAR archives from source, install [7-Zip](https://www.7-zip.org/) and make sure `7z.exe` is in its default installation directory or on `PATH`. Release packages already include the required files.

Download the remaining models from the [model release](https://github.com/Cec1c/Aletheia-Lens/releases/tag/models-v1) and place them at these paths:

| File path | Purpose | Source |
| --- | --- | --- |
| `models/mrcnn/weights.onnx` | Covered-region detection | Model release |
| `models/esrgan/4x-Fatal-Pixels.onnx` | ESRGAN upscaling model | Model release |
| `models/esrgan/4x-Fatal-Pixels.onnx.data` | ESRGAN external weights | Model release; keep next to the file above |
| `models/deepcreampy/bar.onnx` | Bar restoration | Included in the repository |
| `models/deepcreampy/mosaic.onnx` | Mosaic restoration | Included in the repository |

Then start the application:

```powershell
python main.py
```

## Usage

### Choose an input

| Input type | Supported content | Behavior |
| --- | --- | --- |
| Single image (“单图片模式”) | PNG, JPEG, BMP, TIFF, static WebP | Processes the selected image and saves a PNG |
| Folder (“文件夹模式”) | Supported images in nested directories | Processes recursively; can preserve the directory structure or flatten the output |
| Archive (“压缩包模式”) | ZIP, 7Z, RAR | Extracts and processes images, preserving the archive's directory structure in the output folder |

**For folder input, the output directory must be outside the input directory.** Input paths, output paths, and logs support Japanese characters.

### Choose a restoration mode

| Mode | Function | Recommended use |
| --- | --- | --- |
| Mode I (“模式I”) | Automatic bar restoration | The first choice for images covered by bars or solid blocks |
| Mode II (“模式II”) | Automatic mosaic restoration | For pixelated regions; preserves the alpha channel |
| Mode III (“模式III”) | Mosaic restoration and upscaling | Removes transparency; use with care for game assets. Heavy pixelation may produce poor results |

Results depend on the source image and model capabilities. Heavily covered images are more likely to produce distorted results.

### Screentone preprocessing

For scanned manga with dense printed dot patterns, enable “处理前去除漫画网点” (remove screentones before processing). It is off by default. Available strengths are light (“轻度”), medium (“中度”), and strong (“强度”); medium is the default strength.

Preprocessing runs before modes I / II / III and preserves the alpha channel. Higher strength usually removes more dots but may also remove lines and texture. Leave it disabled for images without visible screentones.

### Output files

- Automatic single-image processing saves `processed_<original-filename>.png`, with an additional `.descreen-<strength>` suffix when screentone removal is enabled.
- Folder and archive processing with preserved structure writes to `after_<name>_<short-hash>` and appends `.processed.png` to each complete original filename.
- Flattened output uses stable SHA-256 filenames to prevent name collisions between directories and tasks.
- Screentone strength is included in the output naming or directory hash to distinguish processing settings.

<details>
<summary>Batch output and archive limits</summary>

Output directories with preserved structure are derived from the full source path and processing settings. Flattened filenames are derived from the input root, source-relative path, and processing settings. Settings include the restoration mode and, when enabled, screentone strength.

Each archive is limited to 20,000 members and 20 GiB of extracted data. Before extraction, the application rejects absolute paths, parent-directory traversal, Windows device names, conflicting normalized destinations, symbolic links, directory junctions, and RAR redirections. It also checks available space for temporary files.

</details>

## Experimental features

> [!WARNING]
> The manual painting feature runs, but it is very difficult to use. This project has finally reached the point where it needs a refactor.

### Manual mask editing and restoration

Open **Single image + Mode I → “高级功能（实验性功能）” (Advanced features, experimental) → “手动标注修复” (Manual mask restoration)**.

1. Select an image and output folder. In the editor, use “自动检测” (automatic detection) first, or paint missed regions directly with the brush.
2. The translucent green overlay shows the selection. Erase false positives, or preview and apply “扩大选区” (expand selection). Automatic detection replaces the current selection and can be undone.
3. Click “按当前标注修复” (repair using the current mask). Switch between the working image and result to inspect the output; return to the mask to make adjustments.
4. Click “另存结果” (save result as) to create `manual_<original-filename>.png`. Existing names receive a numeric suffix, preserving the source and previous results.

| Action | Controls |
| --- | --- |
| Zoom | Mouse wheel or plus/minus buttons; “适应窗口” fits the full image |
| Pan | Drag with the right mouse button, or select “平移” (pan) |
| Undo / redo | Buttons, or `Ctrl+Z` / `Ctrl+Y` |
| Edit the selection | Brush, eraser, brush size, clear, expansion preview, and apply |
| Compare views | Switch between “标注” (mask), “工作图” (working image), and “修复结果” (result) |

Brush size and expansion radius use original-image pixels. Each repair starts from the current session's working image, changes colors only inside the selection, and preserves transparency. After changing the mask, run repair again before saving. When screentone removal is enabled, it runs once when the editor opens; the output name includes `.descreen-<strength>`.

Only static single images and mode I are supported. Input and processing settings are fixed for the editing session. You can close the editor during detection or repair, but the background task continues; you must wait for it to finish before starting another processing task. Batch review, model replacement, and broader refactoring are deferred.

## Development and testing

After installing the dependencies and models, run:

```powershell
python -m unittest discover -s tests -v
python main.py --runtime-smoke-test
```

To verify the desktop interface and manual restoration workflow locally:

```powershell
python tools/verify_manual_mask.py
```

This script uses synthetic images, programmatic interactions, and real models. Reports and screenshots are written to `.verification/manual-mask/`. It requires a working desktop graphics environment.

The application is packaged with PyInstaller; see [main.spec](main.spec). The CPU / CUDA12 build and release pipeline is defined in the [GitHub Actions workflow](.github/workflows/build-and-release.yml).

## Credits

| Project or resource | Use |
| --- | --- |
| [DeepCreamPy](https://github.com/cookieY/DeepCreamPy) | Covered-region restoration |
| [hent-AI](https://github.com/natethegreate/hent-AI) | Covered-region detection |
| [4x-Fatal-Pixels](https://openmodeldb.info/models/4x-Fatal-Pixels) | ESRGAN upscaling; this project uses an ONNX conversion |
| [Screentone-Remover](https://github.com/natethegreate/Screentone-Remover) | Reference for the preprocessing approach; the filter code in this project is independently implemented |
| [素材集市康康体](http://www.sucaijishi.com/font-37-792-1.html) | Chinese interface font |
| [deepcreampy-fastapi](https://github.com/fajlkdsjfajdf/deepcreampy-fastapi) | Processing workflow reference |
| [Noto Sans CJK JP](https://github.com/notofonts/noto-cjk) | Japanese paths and logs; the SIL Open Font License 1.1 is bundled with the font |

## Report an issue

For bugs or feature requests, open an [issue](https://github.com/Cec1c/Aletheia-Lens/issues). Include the application version, CPU or CUDA package, input type, restoration mode, and logs to help diagnose the problem.

QQ community group: **829569018**.

## License

This project is licensed under the [GNU GPL v3](LICENSE). Models, fonts, and third-party components remain subject to their respective licenses.
