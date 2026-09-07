"""Opt-in local GUI + real-model smoke test using a synthetic non-content image.

Run from the repository root:
    al-venv\Scripts\python.exe tools\verify_manual_mask.py

Creates PNG screenshots, outputs and a JSON report in .verification/manual-mask.
Pointer observations are injected into the real editor; no desktop input is sent.
"""

import json
from pathlib import Path
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image, ImageDraw

import main


def verify():
    output = ROOT / ".verification" / "manual-mask"
    output.mkdir(parents=True, exist_ok=True)
    source = output / "テスト_透明图片.png"
    image = Image.new("RGBA", (640, 480), (230, 226, 214, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 639, 40), fill=(0, 0, 0, 0))
    draw.rectangle((30, 80, 130, 180), fill=(0, 255, 0, 255))
    draw.ellipse((260, 80, 450, 280), fill=(119, 163, 210, 255))
    draw.rectangle((275, 170, 440, 195), fill=(0, 0, 0, 255))
    draw.rectangle((165, 340, 465, 357), fill=(0, 0, 0, 255))
    image.save(source)
    original = source.read_bytes()
    report = {"source": str(source), "pointer_input": "injected Dear PyGui observations", "checks": []}
    app = main.DeepCreampyApp()
    editor = None
    try:
        dpg.configure_app(manual_callback_management=True)
        dpg.show_viewport()
        dpg.set_primary_window("主窗口", True)

        def frame():
            app.dispatch_ui_callbacks()
            app.tick_manual_editor()
            dpg.render_dearpygui_frame()
            app.adapt_main_window()

        def until(condition, timeout=120):
            deadline = time.monotonic() + timeout
            while not condition():
                if time.monotonic() > deadline:
                    raise TimeoutError("GUI/model task timed out")
                frame()
            for _ in range(3):
                frame()

        def capture(name):
            target = output / name
            target.unlink(missing_ok=True)
            dpg.output_frame_buffer(file=str(target))
            until(target.exists, timeout=10)

        until(lambda: main.PROCESSER_AVAILABLE)
        # Check that only the advanced section exposes this workflow.
        parent = dpg.get_item_parent("手动标注入口")
        assert dpg.get_item_label(parent) == "高级功能（实验性功能）"
        assert dpg.get_value(parent) is False
        for _ in range(8):
            frame()
        collapsed_height = dpg.get_viewport_height()
        dpg.set_value(parent, True)
        dpg.set_value("输入路径", str(source))
        dpg.set_value("输出路径", str(output / "results"))
        for _ in range(8):
            frame()
        assert dpg.get_value("实验性功能警告") == main.EXPERIMENTAL_MANUAL_WARNING
        assert dpg.get_viewport_height() > collapsed_height
        report["checks"].append("expanding experimental section resizes the native window")
        original_size = (dpg.get_viewport_width(), dpg.get_viewport_height())
        original_text_size = dpg.get_text_size("一般情况下如大部分打黑条的本子请选模式I", font=app.ui_font)
        original_font_scale = dpg.get_global_font_scale()
        capture("01-advanced.png")
        with mock.patch.object(main, "viewport_work_area", return_value=(0, 0, 1280, 720)):
            app.fit_viewport(original_size[0], original_size[1])
            for _ in range(8):
                frame()
            assert dpg.get_viewport_height() == 720
            assert dpg.get_y_scroll_max("主窗口") > 0
            dpg.set_y_scroll("主窗口", dpg.get_y_scroll_max("主窗口"))
            for _ in range(4):
                frame()
            assert dpg.is_item_visible("日志输出")
        app.fit_viewport(*original_size)
        dpg.set_y_scroll("主窗口", 0)
        for _ in range(8):
            frame()
        report["checks"].append("short screens retain scrolling to the bottom controls")
        app.open_manual_editor()
        editor = app.manual_editor
        until(lambda: not editor.busy)
        assert dpg.get_item_font(editor.tag("window")) == app.ui_font
        assert dpg.get_item_font(editor.tag("filename")) == app.path_font
        assert editor.document is not None
        assert not editor.document.mask.any()
        assert not dpg.get_item_configuration(editor.tag("repair"))["enabled"]
        report["checks"].append("empty mask disables repair")

        started = time.monotonic()
        editor.detect()
        assert not dpg.get_item_configuration(editor.tag("detect"))["enabled"]
        until(lambda: not editor.busy)
        report["detection_seconds"] = round(time.monotonic() - started, 3)
        report["detected_pixels"] = int(editor.document.mask.sum())
        editor.clear()

        def pointer(point, left=False, clicked=False, right=False):
            origin = dpg.get_item_rect_min(editor.tag("canvas"))
            with (
                mock.patch.object(dpg, "get_mouse_pos", return_value=(origin[0] + point[0], origin[1] + point[1])),
                mock.patch.object(dpg, "is_item_hovered", return_value=True),
                mock.patch.object(dpg, "is_mouse_button_down", side_effect=lambda button: left if button == 0 else right if button == 1 else False),
                mock.patch.object(dpg, "is_mouse_button_clicked", return_value=clicked),
            ):
                editor.tick()
            dpg.render_dearpygui_frame()

        def stroke(start, end):
            t = editor.transform
            p = lambda xy: (xy[0] * t.scale + t.offset_x, xy[1] * t.scale + t.offset_y)
            pointer(p(start), left=True, clicked=True)
            pointer(p(end), left=True)
            pointer(p(end))

        dpg.set_value(editor.tag("brush"), 28)
        stroke((278, 183), (438, 183))
        assert editor.document.mask[183, 350]
        assert not editor.document.mask[120, 80]
        first_stroke = editor.document.mask.copy()
        editor._edit("undo")
        assert not editor.document.mask.any()
        editor._edit("redo")
        np.testing.assert_array_equal(editor.document.mask, first_stroke)
        report["checks"].append("draw stroke and undo/redo through editor callbacks")
        editor.zoom(2)
        pointer((350, 250), right=True)
        pointer((380, 280), right=True)
        pointer((380, 280))
        stroke((325, 230), (345, 230))
        assert editor.document.mask[230, 335]
        editor._edit("undo")
        np.testing.assert_array_equal(editor.document.mask, first_stroke)
        report["checks"].append("zoomed and panned drawing maps to full-resolution mask")
        editor.fit()
        dpg.set_value(editor.tag("tool"), "橡皮")
        stroke((350, 183), (355, 183))
        assert not editor.document.mask[183, 350]
        editor._edit("undo")
        dpg.set_value(editor.tag("tool"), "画笔")
        dpg.set_value(editor.tag("expand"), 2)
        editor._preview_expansion()
        editor.apply_expansion()
        assert editor.document.mask.sum() > first_stroke.sum()
        for _ in range(3):
            frame()
        capture("02-editor.png")

        # Exercise resize and raw-texture replacement with live GPU resources.
        dpg.configure_item(editor.tag("window"), width=850, height=700)
        for _ in range(4):
            frame()
        assert editor.texture_size[0] == 818
        capture("03-resized.png")
        dpg.configure_item(editor.tag("window"), width=1000, height=790)
        for _ in range(4):
            frame()
        report["checks"].append("resize recreates canvas resources")
        mask = editor.document.mask.copy()
        started = time.monotonic()
        editor.repair()
        until(lambda: not editor.busy)
        report["repair_seconds"] = round(time.monotonic() - started, 3)
        assert editor.result is not None, dpg.get_value(editor.tag("status"))
        actual = np.asarray(editor.result)
        expected = np.asarray(image)
        np.testing.assert_array_equal(actual[~mask], expected[~mask])
        np.testing.assert_array_equal(actual[:, :, 3], expected[:, :, 3])
        assert np.any(actual[mask, :3] != expected[mask, :3])
        report["checks"].append("real CPU repair changes selected pixels and exactly preserves outside/alpha")
        capture("04-result.png")
        editor.save()
        until(lambda: not editor.busy)
        saved = set((output / "results").glob("*.png"))
        editor.save()
        until(lambda: not editor.busy)
        assert len(set((output / "results").glob("*.png")) - saved) == 1
        assert source.read_bytes() == original
        report["checks"].append("two saves create distinct files and preserve input")
        editor.clear()
        assert editor.result is None
        assert not dpg.get_item_configuration(editor.tag("save"))["enabled"]

        # An abandoned job remains busy until it finishes, without GUI access.
        import threading
        gate = threading.Event()
        editor._submit("repair", lambda: (gate.wait(5), image.copy())[1], "测试关闭")
        editor.close()
        gate.set()
        until(lambda: not editor.busy)
        assert editor.result is None
        assert (dpg.get_viewport_width(), dpg.get_viewport_height()) == original_size
        assert dpg.get_global_font_scale() == original_font_scale
        assert dpg.get_item_font("主窗口") == app.ui_font
        assert dpg.get_text_size("一般情况下如大部分打黑条的本子请选模式I", font=app.ui_font) == original_text_size
        capture("05-returned.png")
        with Image.open(output / "01-advanced.png") as before, Image.open(output / "05-returned.png") as after:
            np.testing.assert_array_equal(np.asarray(before)[:150], np.asarray(after)[:150])
        report["checks"].append("return restores viewport geometry and preserves the main UI font and size")
        report["checks"].append("closed editor safely discards delayed completion")
        app.open_manual_editor()
        editor = app.manual_editor
        until(lambda: not editor.busy)
        assert not editor.document.mask.any()
        for index in range(2):
            editor.close()
            for _ in range(8):
                frame()
            assert dpg.get_item_font("主窗口") == app.ui_font
            assert (dpg.get_viewport_width(), dpg.get_viewport_height()) == original_size
            if index == 0:
                app.open_manual_editor()
                editor = app.manual_editor
                until(lambda: not editor.busy)
        report["checks"].append("three open/close cycles preserve font and viewport")
        report["checks"].append("reopening creates a fresh annotation session")
        report["status"] = "PASS"
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if editor is not None:
            editor.close()
        dpg.destroy_context()


if __name__ == "__main__":
    verify()
