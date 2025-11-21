import dearpygui.dearpygui as dpg
import os
import threading
from tkinter import filedialog
import tkinter as tk


# 初始设置为False
PROCESSER_AVAILABLE = False
process_image_stream = None  # 全局变量存储处理函数

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
        
        # 初始化Dear PyGui
        dpg.create_context()
        dpg.create_viewport(title='Aletheia Lens', small_icon='ico.ico', large_icon='ico.ico', width=600, height=780)
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
                "./font/sckkt.ttf",  # 用户指定的字体
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
        esrgan_model_path = os.path.join("models", "esrgan", "4x-Fatal-Pixels.pth")
        if os.path.exists(esrgan_model_path):
            dpg.set_value("放大模型状态", "可用")
            dpg.configure_item("放大模型状态", color=(0, 255, 0, 255))
        else:
            dpg.set_value("放大模型状态", "不可用")
            dpg.configure_item("放大模型状态", color=(255, 0, 0, 255))
        
        # 检查检测模型
        mrcnn_model_path = os.path.join("models", "mrcnn", "weights.h5")
        if os.path.exists(mrcnn_model_path):
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
                        items=["单图片模式", "文件夹模式"],
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
        self.input_type = "image" if app_data == "单图片模式" else "folder"
        
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
    
    def process_with_structure(self, input_dir, output_base_dir, image_files):
        """处理文件夹并保留结构"""
        # 创建顶层输出文件夹（添加after_前缀）
        input_folder_name = os.path.basename(input_dir)
        output_top_folder = f"after_{input_folder_name}"
        output_dir = os.path.join(output_base_dir, output_top_folder)
        
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
            output_path = os.path.join(output_relative_dir, filename)  # 保持原文件名
            
            # 确保输出目录存在
            os.makedirs(output_relative_dir, exist_ok=True)
            
            try:
                # 处理图片
                with open(input_path, "rb") as f:
                    image_bytes = f.read()
                
                # 使用全局的process_image_stream函数处理图片
                result_image = process_image_stream(image_bytes, self.mode)
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
            
            # 生成唯一的输出文件名（避免重名）
            if relative_path == ".":
                # 根目录文件
                output_filename = filename
            else:
                # 子目录文件，在文件名前添加路径信息避免冲突
                safe_path = relative_path.replace(os.path.sep, "_")
                name, ext = os.path.splitext(filename)
                output_filename = f"{safe_path}_{name}{ext}"
            
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                # 处理图片
                with open(input_path, "rb") as f:
                    image_bytes = f.read()
                
                # 使用全局的process_image_stream函数处理图片
                result_image = process_image_stream(image_bytes, self.mode)
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

def main():
    """主函数"""
    app = DeepCreampyApp()
    app.run()

if __name__ == "__main__":
    main()