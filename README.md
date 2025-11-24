# Aletheia-Lens
一个基于deepcreampy的自动涂抹识别去码工具，带有GUI界面和打包版本，极易使用
## 适用场景

- 我有一批带黑条的图片，只有少部分可猜测细节被遮挡，我希望能批量修复他们并且不影响原有命名顺序
- 我解包了一个游戏，找到了其带码的素材，我想批量转换他们但是素材太多了，分不清其文件结构，而且不想破坏原有素材的图层透明性(模式II不会，模式III会破坏
- 等等场景，作者做来主要是针对游戏解包素材再修复，如有需求可以交个issue我试试
- 仅适用于二次元图片
- 
# 简介
一个使用了deepcreampy和hent-AI的来实现自动涂抹和自动去码的AI工具 

[点我一键下载](https://github.com/Cec1c/Aletheia-Lens/releases/download/withdir/Aletheia-Lens_release_dirV1.0.7z)

~~部署API?~~

~~安装Python环境？~~

现在一次性帮你打包好了，当然，如果你不打算下载我打包的release版本使用的话，你仍然需要按照下文的办法手动安装模型
# 工具GUI
![alt text](https://github.com/Cec1c/Aletheia-Lens/blob/main/readmeimg/%E5%B7%A5%E5%85%B7GUI.png)
# 安装办法
### 即开即用版本
release版本直接 [点击我](https://github.com/Cec1c/Aletheia-Lens/releases/download/withdir/Aletheia-Lens_release_dirV1.0.7z) 

即可下载，下载后解压打开Aletheia-Lens.exe 直接运行即可

程序使用pyinstaller打包，若需要自行打包可以下载源码看下边python安装方法

（我去pyinstaller你这打包下来是真夸张啊）
### 已有python不想安装那么多东西，你这也太大了

项目下载下来或者克隆下来打开项目根目录打开cmd输入
``` 
pip install -r requirements.txt 
```
安装完成后（你需要先把下边模型安装齐全）
```
python main.py
```
### 模型安装

如果你是直接下载的release版本你无需关注本部分，为了方便使用我就顺便集成进去了，（侵删）

如果你是python办法下载源代码自行运行的

考虑到用到的deepcream模型是onnx导出过的，这个就集成进models文件夹里了

你需要下载以下两个模型：

<a href="https://github.com/natethegreate/hent-AI">hent-AI</a>(打开后往下找到最新的228步模型打开dropbox网盘，无需登录可以满速下载)  

<a href="https://openmodeldb.info/models/4x-Fatal-Pixels">4x-Fatal-Pixels.pth</a> 

他们分别放到models/mrcnn 和 models/esrgan 文件夹下

注意命名必须对的上

看到程序模型部分全绿就意味着正常打开了

# 使用办法
实际上相当简单啊，我相信大多数人一打开就会了

**另外这个工具只支持二次元图片**

先选择模式，如果只有单图就选单图，如果是在一堆文件夹里面则选择文件夹模式

![alt text](https://github.com/Cec1c/Aletheia-Lens/blob/main/readmeimg/%E6%A8%A1%E5%BC%8F.png)

然后选择输入输出文件夹，输入文件夹就是你要修复的图片所在的文件夹，输出文件夹就是修复后的图片存放的文件夹

输出文件夹会默认保留原先文件夹的结构，同时最顶层文件夹会添加after_后缀

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

还使用了<a href="https://openmodeldb.info/models/4x-Fatal-Pixels">4x-Fatal-Pixels.pth</a> 

前者提供对涂抹部分去码，中者用于识别并涂抹码区，后者用于放大功能

使用了前人 [fastapi](https://github.com/fajlkdsjfajdf/deepcreampy-fastapi) 的调用处理过程，在基础上修改了一些调用，并做了一个GUI窗体

以及使用了免费字体：<a href="http://www.sucaijishi.com/font-37-792-1.html">素材集市康康体</a>

感谢以上

# 报错或者疑难杂症

测试倒是没怎么做，这是个花了三天做出来的简单程序，碰到问题是难免的，可以交个issue

或者其他疑难杂症加入Q群反馈 ：829569018




