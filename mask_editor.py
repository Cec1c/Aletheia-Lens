"""Single-image mask editing, independent of model loading and source colors."""

from dataclasses import dataclass
from functools import lru_cache
import io
from pathlib import Path
import queue
import threading

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image


class MaskDocument:
    """Full-resolution annotations with bounded, bit-packed undo history."""

    def __init__(self, image, history_bytes=32 * 1024 * 1024):
        self.image = image.copy()
        self.mask = np.zeros((image.height, image.width), dtype=np.bool_)
        self.history_limit = max(1, min(40, history_bytes // max(1, (self.mask.size + 7) // 8)))
        self.undo_stack = []
        self.redo_stack = []
        self.stroke_before = None
        self.last_point = None

    def _remember(self, before):
        if np.array_equal(before, np.packbits(self.mask)):
            return False
        self.undo_stack.append(before)
        del self.undo_stack[:-self.history_limit]
        self.redo_stack.clear()
        return True

    def replace(self, mask):
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != self.mask.shape:
            raise ValueError("选区尺寸或类型不正确")
        before = np.packbits(self.mask)
        self.mask[:] = mask
        return self._remember(before)

    def begin_stroke(self):
        self.stroke_before = np.packbits(self.mask)
        self.last_point = None

    def paint(self, point, diameter, erase=False):
        if point is None:
            self.last_point = None
            return False
        x, y = point
        if not (0 <= x < self.image.width and 0 <= y < self.image.height):
            self.last_point = None
            return False
        current = (int(x), int(y))
        if current == self.last_point:
            return False
        value = 0 if erase else 1
        thickness = max(1, int(diameter))
        pixels = self.mask.view(np.uint8)
        if self.last_point is not None:
            cv2.line(pixels, self.last_point, current, value, thickness, cv2.LINE_8)
        cv2.circle(pixels, current, max(0, thickness // 2), value, -1, cv2.LINE_8)
        self.last_point = current
        return True

    def end_stroke(self):
        if self.stroke_before is None:
            return False
        before, self.stroke_before = self.stroke_before, None
        self.last_point = None
        return self._remember(before)

    def undo(self):
        return self._restore(self.undo_stack, self.redo_stack)

    def redo(self):
        return self._restore(self.redo_stack, self.undo_stack)

    def _restore(self, source, destination):
        if not source:
            return False
        destination.append(np.packbits(self.mask))
        self.mask[:] = np.unpackbits(source.pop(), count=self.mask.size).reshape(self.mask.shape)
        return True

    def expanded(self, radius):
        if radius <= 0:
            return self.mask
        radius = int(radius)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        return cv2.dilate(self.mask.view(np.uint8), kernel).astype(np.bool_)


@dataclass
class CanvasTransform:
    image_width: int
    image_height: int
    width: int
    height: int
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def fit(self):
        self.scale = min(self.width / self.image_width, self.height / self.image_height)
        self.offset_x = (self.width - self.image_width * self.scale) / 2
        self.offset_y = (self.height - self.image_height * self.scale) / 2

    def to_image(self, point):
        x, y = point
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        x = (x - self.offset_x) / self.scale
        y = (y - self.offset_y) / self.scale
        if 0 <= x < self.image_width and 0 <= y < self.image_height:
            return x, y
        return None

    def zoom(self, factor, anchor):
        new_scale = min(32.0, max(0.01, self.scale * factor))
        ratio = new_scale / self.scale
        self.offset_x = anchor[0] - (anchor[0] - self.offset_x) * ratio
        self.offset_y = anchor[1] - (anchor[1] - self.offset_y) * ratio
        self.scale = new_scale

    def affine(self):
        return (1 / self.scale, 0, -self.offset_x / self.scale,
                0, 1 / self.scale, -self.offset_y / self.scale)


@lru_cache(maxsize=2)
def _checkerboard(size):
    y, x = np.indices((size[1], size[0]))
    gray = np.where((x // 12 + y // 12) % 2, 56, 69).astype(np.uint8)
    return Image.fromarray(np.dstack((gray, gray, gray, np.full_like(gray, 255))))


def render_canvas(image, mask, transform):
    """Render only the visible canvas; annotations retain their original size."""
    size = (transform.width, transform.height)
    # Cache premultiplied previews in the editor so Pillow need not convert the
    # entire original RGBA image for every affine resampling operation.
    rgba = image if image.mode in ("RGBA", "RGBa") else image.convert("RGBA")
    visible = rgba.transform(size, Image.Transform.AFFINE, transform.affine(), Image.Resampling.BILINEAR).convert("RGBA")
    canvas = Image.alpha_composite(_checkerboard(size), visible)
    if mask is not None:
        # A shared uint8 view avoids copying the full mask on every mouse move.
        # Apply display opacity only after sampling the visible part.
        selection = Image.fromarray(mask.view(np.uint8)).transform(
            size, Image.Transform.AFFINE, transform.affine(), Image.Resampling.NEAREST,
        ).point(lambda value: value * 115)
        overlay = Image.new("RGBA", size, (50, 225, 135, 0))
        overlay.putalpha(selection)
        canvas = Image.alpha_composite(canvas, overlay)
    return np.asarray(canvas, dtype=np.float32).ravel() / 255.0


def save_manual_result(image, source_path, output_dir, screentone_level=0):
    """Create a fresh PNG without overwriting either source or previous results."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = f".descreen-{screentone_level}" if screentone_level else ""
    stem = f"manual_{Path(source_path).stem}{suffix}"
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    source = Path(source_path).resolve()
    index = 0
    while True:
        number = f"-{index}" if index else ""
        target = directory / f"{stem}{number}.png"
        index += 1
        if target.resolve() == source:
            continue
        try:
            stream = target.open("xb")
        except FileExistsError:
            continue
        try:
            with stream:
                stream.write(buffer.getvalue())
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return str(target)


class MaskEditor:
    """Dear PyGui editor. Call tick on the UI thread, once per render frame.

    Workers receive snapshots and return through a queue. A closed editor never
    applies late results to a new session or touches deleted GUI items.
    """

    def __init__(self, source_path, output_dir, screentone_level=0, log=print):
        self.source_path = str(Path(source_path).resolve())
        self.output_dir = str(Path(output_dir).resolve())
        self.screentone_level = screentone_level
        self.log = log
        self.document = None
        self.result = None
        self.preview_image = None
        self.preview_result = None
        self.transform = None
        self.closed = False
        self.busy = False
        self.generation = 0
        self.completions = queue.Queue()
        self.dirty = True
        self.pan_point = None
        self.texture_size = None
        self.prefix = f"mask_{dpg.generate_uuid()}_"
        self._build()

        def prepare():
            from processer import prepare_manual_image
            return prepare_manual_image(Path(self.source_path).read_bytes(), screentone_level)

        self._submit("load", prepare, "正在读取图片并准备标注画面…")

    def tag(self, name):
        return self.prefix + name

    def _build(self):
        with dpg.texture_registry(tag=self.tag("textures")):
            pass
        with dpg.window(label="手动标注修复（实验性功能） / 模式 I", tag=self.tag("window"),
                        width=min(1000, dpg.get_viewport_width() - 40),
                        height=dpg.get_viewport_height() - 70, pos=(12, 12), min_size=(820, 600),
                        modal=True, no_collapse=True, no_scrollbar=True,
                        no_scroll_with_mouse=True, on_close=self.close):
            with dpg.group(tag=self.tag("toolbar")):
                dpg.add_text(Path(self.source_path).name, tag=self.tag("filename"), wrap=930)
                dpg.add_text("先自动检测或直接涂抹，再按当前标注修复；绿色为选区，棋盘格为透明区域。",
                             tag=self.tag("instructions"), wrap=930)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="自动检测（替换选区，可撤销）", tag=self.tag("detect"), callback=self.detect)
                    dpg.add_button(label="撤销", tag=self.tag("undo"), callback=lambda: self._edit("undo"))
                    dpg.add_button(label="重做", tag=self.tag("redo"), callback=lambda: self._edit("redo"))
                    dpg.add_button(label="清空选区", tag=self.tag("clear"), callback=self.clear)
                with dpg.group(horizontal=True):
                    dpg.add_radio_button(items=["画笔", "橡皮", "平移"], horizontal=True,
                                         default_value="画笔", tag=self.tag("tool"))
                    dpg.add_slider_int(label="笔刷（原图像素）", min_value=1, max_value=200,
                                       default_value=20, width=150, tag=self.tag("brush"))
                with dpg.group(horizontal=True):
                    dpg.add_slider_int(label="扩大预览", min_value=0, max_value=20,
                                       default_value=0, width=110, tag=self.tag("expand"), callback=self._preview_expansion)
                    dpg.add_button(label="应用扩大", tag=self.tag("apply"), callback=self.apply_expansion)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="适应窗口", callback=self.fit, tag=self.tag("fit"))
                    dpg.add_button(label="－", callback=lambda: self.zoom(0.8), tag=self.tag("out"))
                    dpg.add_button(label="＋", callback=lambda: self.zoom(1.25), tag=self.tag("in"))
                    dpg.add_radio_button(items=["标注", "工作图", "修复结果"], horizontal=True,
                                         default_value="标注", tag=self.tag("view"), callback=self._view_changed)
                    dpg.add_text("", tag=self.tag("zoom"))
            dpg.add_drawlist(width=968, height=480, tag=self.tag("canvas"))
            with dpg.group(tag=self.tag("footer")):
                dpg.add_text("", tag=self.tag("status"), wrap=940)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="按当前标注修复", tag=self.tag("repair"), callback=self.repair, width=160)
                    dpg.add_button(label="另存结果", tag=self.tag("save"), callback=self.save, width=110)
                    dpg.add_button(label="关闭", callback=self.close)
                dpg.add_text("滚轮缩放 / 右键拖动平移 / Ctrl+Z / Ctrl+Y", tag=self.tag("shortcuts"), wrap=930)
        with dpg.handler_registry(tag=self.tag("handlers")):
            dpg.add_mouse_wheel_handler(callback=self._wheel)
            dpg.add_key_press_handler(key=dpg.mvKey_Z, callback=lambda: self._shortcut("undo"))
            dpg.add_key_press_handler(key=dpg.mvKey_Y, callback=lambda: self._shortcut("redo"))
        self._controls()

    def _status(self, text):
        dpg.set_value(self.tag("status"), text)

    def _controls(self):
        ready = self.document is not None and not self.busy
        for name in ("detect", "clear", "tool", "brush", "expand", "apply", "fit", "out", "in", "view"):
            dpg.configure_item(self.tag(name), enabled=ready)
        dpg.configure_item(self.tag("undo"), enabled=ready and bool(self.document.undo_stack))
        dpg.configure_item(self.tag("redo"), enabled=ready and bool(self.document.redo_stack))
        dpg.configure_item(self.tag("repair"), enabled=ready and bool(self.document.mask.any()))
        dpg.configure_item(self.tag("save"), enabled=ready and self.result is not None)

    def _submit(self, kind, operation, message):
        if self.closed or self.busy:
            return
        self.busy = True
        generation = self.generation
        self._status(message)
        self._controls()

        def worker():
            try:
                value, error = operation(), None
            except Exception as exc:
                value, error = None, str(exc)
            self.completions.put((generation, kind, value, error))

        threading.Thread(target=worker, daemon=True, name="manual-mask-worker").start()

    def _drain(self):
        try:
            generation, kind, value, error = self.completions.get_nowait()
        except queue.Empty:
            return
        self.busy = False
        if self.closed or generation != self.generation:
            return
        if error is not None:
            self._status(f"操作失败：{error}；可修改后重试。")
            self.log(f"手动标注 {kind} 失败：{error}")
        elif kind == "load":
            self.document = MaskDocument(value)
            self.preview_image = value.convert("RGBa")
            self._status("图片已就绪。可自动检测，也可直接画选区；当前没有选区。")
        elif kind == "detect":
            self.document.replace(value)
            self._changed()
            self._status("检测完成，可继续补标或擦除。" if value.any() else "未检测到黑条；可以直接用画笔标注。")
        elif kind == "repair":
            self.result = value
            self.preview_result = value.convert("RGBa")
            dpg.set_value(self.tag("view"), "修复结果")
            self._status("修复完成。切回“标注”可继续调整；“另存结果”保存新文件。")
        elif kind == "save":
            self._status(f"已保存：{value}")
            self.log(f"手动标注结果已保存：{value}")
        self.dirty = True
        self._controls()

    def _changed(self):
        self.result = None
        self.preview_result = None
        self.dirty = True
        dpg.set_value(self.tag("view"), "标注")
        self._controls()

    def _edit(self, method):
        if self.busy or self.document is None:
            return
        dpg.set_value(self.tag("expand"), 0)
        if getattr(self.document, method)():
            self._changed()
            self._status("选区已更新；修复时将重新使用本次工作图。")

    def clear(self):
        if not self.busy and self.document is not None:
            dpg.set_value(self.tag("expand"), 0)
            if self.document.replace(np.zeros_like(self.document.mask)):
                self._changed()
            self._status("选区已清空，可撤销。")

    def _preview_expansion(self):
        self.dirty = True
        dpg.set_value(self.tag("view"), "标注")
        # The result predates the pending selection, so it cannot be saved.
        self.result = None
        self.preview_result = None
        self._controls()
        self._status("正在预览扩大选区；点击“应用扩大”确认，或将预览值调回 0。修复也会应用当前预览。")

    def apply_expansion(self):
        if self.busy or self.document is None:
            return
        radius = dpg.get_value(self.tag("expand"))
        if radius:
            changed = self.document.replace(self.document.expanded(radius))
            dpg.set_value(self.tag("expand"), 0)
            if changed:
                self._changed()
                self._status("已应用扩大选区，可撤销；可以继续标注或执行修复。")

    def detect(self):
        if self.busy or self.document is None:
            return
        image = self.document.image.copy()
        dpg.set_value(self.tag("expand"), 0)

        def operation():
            from processer import detect_manual_bars
            return detect_manual_bars(image)

        self._submit("detect", operation, "正在自动检测… 完成后会替换当前选区，可撤销。")

    def repair(self):
        if self.busy or self.document is None:
            return
        self.apply_expansion()
        if not self.document.mask.any():
            self._status("请先用画笔标注需要修复的区域。")
            return
        image, mask = self.document.image.copy(), self.document.mask.copy()

        def operation():
            from processer import repair_manual_bars
            return repair_manual_bars(image, mask)

        self._submit("repair", operation, "正在按当前标注修复… CPU 处理可能需要一些时间。")

    def save(self):
        if self.busy or self.result is None:
            return
        image = self.result.copy()
        self._submit("save", lambda: save_manual_result(image, self.source_path, self.output_dir,
                                                        self.screentone_level), "正在保存新文件…")

    def _view_changed(self):
        if dpg.get_value(self.tag("view")) == "修复结果" and self.result is None:
            dpg.set_value(self.tag("view"), "标注")
            self._status("尚无修复结果，请先按当前标注修复。")
        self.dirty = True

    def fit(self):
        if self.transform is not None:
            self.transform.fit()
            self.dirty = True

    def zoom(self, factor, anchor=None):
        if self.transform is not None:
            self.transform.zoom(factor, anchor or (self.transform.width / 2, self.transform.height / 2))
            self.dirty = True

    def _mouse(self):
        origin = dpg.get_item_rect_min(self.tag("canvas"))
        position = dpg.get_mouse_pos(local=False)
        return position[0] - origin[0], position[1] - origin[1]

    def _wheel(self, sender, amount):
        if not self.closed and not self.busy and self.transform is not None and dpg.is_item_hovered(self.tag("canvas")):
            self.zoom(1.2 ** amount, self._mouse())

    def _shortcut(self, operation):
        if not self.closed and dpg.is_key_down(dpg.mvKey_Control):
            self._edit(operation)

    def _pointer(self):
        point = self._mouse()
        hovered = dpg.is_item_hovered(self.tag("canvas"))
        left = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        right = dpg.is_mouse_button_down(dpg.mvMouseButton_Right)
        tool = dpg.get_value(self.tag("tool"))
        pan = right or (left and tool == "平移")
        if pan:
            if self.pan_point is not None:
                self.transform.offset_x += point[0] - self.pan_point[0]
                self.transform.offset_y += point[1] - self.pan_point[1]
                self.dirty = True
            if hovered or self.pan_point is not None:
                self.pan_point = point
        else:
            self.pan_point = None
        if left and not pan and dpg.get_value(self.tag("view")) == "标注":
            if hovered and dpg.is_mouse_button_clicked(dpg.mvMouseButton_Left):
                self.apply_expansion()
                self.document.begin_stroke()
            if self.document.stroke_before is not None:
                changed = self.document.paint(self.transform.to_image(point) if hovered else None,
                                              dpg.get_value(self.tag("brush")), erase=tool == "橡皮")
                self.dirty = self.dirty or changed
        elif self.document.end_stroke():
            self._changed()
        if dpg.does_item_exist(self.tag("cursor")):
            dpg.delete_item(self.tag("cursor"))
        if hovered and tool != "平移" and dpg.get_value(self.tag("view")) == "标注" and self.transform.to_image(point) is not None:
            radius = max(1, dpg.get_value(self.tag("brush")) * self.transform.scale / 2)
            dpg.draw_circle(point, radius, color=(255, 255, 255, 220), thickness=1,
                            parent=self.tag("canvas"), tag=self.tag("cursor"))

    def tick(self):
        self._drain()
        if self.closed or self.document is None:
            return
        width, height = dpg.get_item_rect_size(self.tag("window"))
        toolbar_height = dpg.get_item_rect_size(self.tag("toolbar"))[1]
        footer_height = dpg.get_item_rect_size(self.tag("footer"))[1]
        size = (max(320, int(width) - 32), max(120, int(height - toolbar_height - footer_height - 50)))
        if size != self.texture_size:
            for name in ("status", "filename", "instructions", "shortcuts"):
                dpg.configure_item(self.tag(name), wrap=size[0])
            dpg.configure_item(self.tag("canvas"), width=size[0], height=size[1])
            dpg.delete_item(self.tag("canvas"), children_only=True)
            if dpg.does_item_exist(self.tag("texture")):
                dpg.delete_item(self.tag("texture"))
            self.texture_data = np.zeros(size[0] * size[1] * 4, dtype=np.float32)
            dpg.add_raw_texture(*size, default_value=self.texture_data, format=dpg.mvFormat_Float_rgba,
                                tag=self.tag("texture"), parent=self.tag("textures"))
            dpg.draw_image(self.tag("texture"), (0, 0), size, parent=self.tag("canvas"))
            if self.transform is None:
                self.transform = CanvasTransform(*self.document.image.size, *size)
                self.transform.fit()
            else:
                self.transform.offset_x += (size[0] - self.transform.width) / 2
                self.transform.offset_y += (size[1] - self.transform.height) / 2
                self.transform.width, self.transform.height = size
            self.texture_size = size
            self.dirty = True
        if not self.busy:
            self._pointer()
        if self.dirty:
            view = dpg.get_value(self.tag("view"))
            image = self.preview_result if view == "修复结果" and self.result is not None else self.preview_image
            mask = self.document.expanded(dpg.get_value(self.tag("expand"))) if view == "标注" else None
            self.texture_data[:] = render_canvas(image, mask, self.transform)
            dpg.set_value(self.tag("zoom"), f"{self.transform.scale:.0%} / {self.document.image.width} × {self.document.image.height}")
            self.dirty = False

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.generation += 1
        for name in ("handlers", "window", "textures"):
            dpg.delete_item(self.tag(name))
