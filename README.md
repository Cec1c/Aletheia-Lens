# Aletheia-Lens
一个基于deepcreampy的自动涂抹识别去码工具，带有GUI界面和打包版本，极易使用
## 适用场景

- 我有一批带黑条的图片，只有少部分可猜测细节被遮挡，我希望能批量修复他们并且不影响原有命名顺序
- 我解包了一个游戏，找到了其带码的素材，我想批量转换他们但是素材太多了，分不清其文件结构，而且不想破坏原有素材的图层透明性(模式II不会，模式III会破坏)
- 等等场景，作者做来主要是针对游戏解包素材再修复，如有需求可以交个issue我试试
- 仅适用于二次元图片
# 简介
一个使用了deepcreampy和hent-AI的来实现自动涂抹和自动去码的AI工具 

[前往 Release 页面下载](https://github.com/Cec1c/Aletheia-Lens/releases/latest)

~~部署API?~~

~~安装Python环境？~~

现在一次性帮你打包好了，当然，如果你不打算下载我打包的release版本使用的话，你仍然需要按照下文的办法手动安装模型
# 工具GUI
![alt text](https://github.com/Cec1c/Aletheia-Lens/blob/main/readmeimg/%E5%B7%A5%E5%85%B7GUI.png)
# 安装办法
### 即开即用版本

每个新版本提供两个压缩包：

- `*_cpu.7z`：体积较小，使用 CPU。
- `*_cuda12.7z`：NVIDIA GPU 版，内置 CUDA 12 / cuDNN 9 运行库。

下载后解压并运行 `Aletheia-Lens.exe`。程序日志会明确显示 ONNX 会话已启用 CUDA、混合状态还是纯 CPU。CUDA 包会让 Mask R-CNN 检测与 ESRGAN 放大优先使用 GPU；DeepCreamPy 的 `bar.onnx` / `mosaic.onnx` 在 CUDA12/cuDNN9 下会产生非有限值，因此明确固定到 CPU，日志会说明这是兼容性策略而不是静默回退。

程序使用pyinstaller打包，若需要自行打包可以下载源码看下边python安装方法

（我去pyinstaller你这打包下来是真夸张啊）
### 已有python不想安装那么多东西，你这也太大了

这个项目构建在**Python 3.10.11**下，尚不清楚其他版本是否会出现问题，如遇兼容性问题可告知我我在这留下信息

CPU 环境：

```
pip install -r requirements.txt
```

NVIDIA GPU 环境：

```
pip install -r requirements-gpu.txt
```
源码运行 RAR 压缩包还需要安装 7-Zip，并确保 `7z.exe` 位于默认安装目录或 `PATH`；打包版已经内置所需文件。

安装完成后（你需要先把下边模型安装齐全）
```
python main.py
```
### 模型安装

如果你是直接下载的release版本你无需关注本部分，为了方便使用我就顺便集成进去了，（侵删）

如果你是python办法下载源代码自行运行的

考虑到用到的deepcream模型是onnx导出过的，这个就集成进models文件夹里了

运行时需要以下 ONNX 文件：

- `models/mrcnn/weights.onnx`
- `models/esrgan/4x-Fatal-Pixels.onnx`
- `models/esrgan/4x-Fatal-Pixels.onnx.data`

DeepCreamPy 的 `bar.onnx` 和 `mosaic.onnx` 已包含在仓库中。其余模型由项目的模型 Release 提供。

注意命名必须对的上

看到程序模型部分全绿就意味着正常打开了

# 使用办法
实际上相当简单啊，我相信大多数人一打开就会了

**另外这个工具只支持二次元图片**

先选择输入类型：单图片、文件夹或压缩包模式。图片支持 PNG、JPEG、BMP、TIFF 和静态 WebP；压缩包模式支持 ZIP、7Z 和 RAR，处理后保留包内目录结构；保留结构的文件夹和压缩包会按“源完整路径 + 处理模式”创建独立的 `after_<名称>_<短哈希>` 目录，结果文件在完整原文件名后追加 `.processed.png`。平铺模式使用输入根、源相对路径和处理模式生成稳定 SHA-256 名称，避免目录压平或跨任务复用输出目录时同名覆盖。文件夹输出目录必须位于输入目录之外。

为避免压缩炸弹或链接越界，单个压缩包最多包含 20,000 个成员、解压后总大小最多 20 GiB。解压前会拒绝绝对路径、父目录跳转、Windows 设备名、规范化后冲突的目标，以及符号链接、目录联接和 RAR 重定向，并检查临时目录剩余空间。

![alt text](https://github.com/Cec1c/Aletheia-Lens/blob/main/readmeimg/%E6%A8%A1%E5%BC%8F.png)

然后选择输入输出文件夹，输入文件夹就是你要修复的图片所在的文件夹，输出文件夹就是修复后的图片存放的文件夹

输出文件夹会默认保留原先文件夹的结构，同时最顶层使用 `after_<名称>_<短哈希>` 区分不同输入和处理模式。

![alt text](https://github.com/Cec1c/Aletheia-Lens/blob/main/readmeimg/%E9%80%89%E6%8B%A9%E6%96%87%E4%BB%B6%E5%A4%B9.png)

接着选择模式，对于大部分本子用的黑条去码，选择模式I，对色块的漫画去码效果应该是不赖的

对于一般的马赛克选择模式II

模式III不是很建议使用，出来的效果有时很诡异

![alt text](https://github.com/Cec1c/Aletheia-Lens/blob/main/readmeimg/%E5%A4%84%E7%90%86%E6%A8%A1%E5%BC%8F.png)

接着一切就绪，点击开始处理即可

需要注意的是，执行前请确保模块都加载好了

![alt text](https://github.com/Cec1c/Aletheia-Lens/blob/main/readmeimg/%E6%A8%A1%E5%9E%8B%E7%8A%B6%E6%80%81.png)

# 工具说明

本工具基于<a href="https://github.com/cookieY/DeepCreamPy">deepcreampy</a> 和 <a href="https://github.com/natethegreate/hent-AI">hent-AI</a> 

还使用了<a href="https://openmodeldb.info/models/4x-Fatal-Pixels">4x-Fatal-Pixels</a> 的 ONNX 转换模型。

前者提供对涂抹部分去码，中者用于识别并涂抹码区，后者用于放大功能

使用了前人 [fastapi](https://github.com/fajlkdsjfajdf/deepcreampy-fastapi) 的调用处理过程，在基础上修改了一些调用，并做了一个GUI窗体

以及使用了免费字体：<a href="http://www.sucaijishi.com/font-37-792-1.html">素材集市康康体</a>。路径和日志中的日文字符由 <a href="https://github.com/notofonts/noto-cjk">Noto Sans CJK JP</a> 提供，该字体使用 SIL Open Font License 1.1，许可证随字体一起打包。

感谢以上

# 报错或者疑难杂症

项目包含 ONNX、CUDA 打包、压缩包安全和批量输出的自动化回归测试；如仍遇到问题可以提交 issue。

或者其他疑难杂症加入Q群反馈 ：829569018




