import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import dearpygui.dearpygui as dpg
from PIL import Image, ImageFont

import image_formats
import main
from tools import image_tool


ROOT = Path(__file__).resolve().parents[1]
NOTO_FONT = ROOT / "font" / "NotoSansCJKjp-Regular.otf"
NOTO_LICENSE = ROOT / "font" / "NotoSansCJK-LICENSE.txt"
NOTO_FONT_SHA256 = "68a3fc98800b2a27b371f2fb79991daf3633bd89309d4ffaa6946fd587f375b5"


def _webp_bytes(color="red"):
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(output, format="WEBP")
    return output.getvalue()


class WebpAndUnicodePathTests(unittest.TestCase):
    def setUp(self):
        self.app = main.DeepCreampyApp.__new__(main.DeepCreampyApp)
        self.app.processing = True
        self.app.mode = 1
        self.app.current_file_index = 0
        self.app.log_message = mock.Mock()
        self.app.update_progress = mock.Mock()
        self.app.log_runtime_status = mock.Mock()

    def test_webp_is_declared_once_for_all_entry_points(self):
        self.assertIn(".webp", image_formats.SUPPORTED_IMAGE_EXTENSIONS)
        self.assertIn("*.webp", image_formats.IMAGE_FILE_DIALOG_PATTERN)
        self.assertTrue(image_formats.is_supported_image("画像.WEBP"))

    def test_single_image_picker_offers_webp(self):
        self.app.input_type = "image"
        with (
            mock.patch.object(main.tk, "Tk"),
            mock.patch.object(main.filedialog, "askopenfilename", return_value="") as picker,
        ):
            self.app.browse_input()

        file_pattern = picker.call_args.kwargs["filetypes"][0][1]
        self.assertIn("*.webp", file_pattern)

    def test_runtime_image_decoder_accepts_static_webp(self):
        decoded = image_tool.bytes2npimage(_webp_bytes())

        self.assertEqual(decoded.shape, (8, 8, 3))

    def test_single_webp_with_japanese_path_is_processed(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            input_dir = root / "入力_日本語"
            output_dir = root / "出力_日本語"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "画像_テスト.webp"
            source.write_bytes(_webp_bytes())

            self.app.input_path = str(source)
            self.app.output_path = str(output_dir)
            processed = Image.new("RGB", (8, 8), color="blue")
            with mock.patch.object(main, "process_image_stream", return_value=processed):
                self.app.process_single_file()

            output = output_dir / "processed_画像_テスト.png"
            self.assertTrue(output.exists())
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_folder_scans_and_processes_webp_with_japanese_paths(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            input_dir = root / "入力_日本語"
            chapter = input_dir / "第１章_かなカナ"
            output_dir = root / "出力_日本語"
            chapter.mkdir(parents=True)
            output_dir.mkdir()
            (chapter / "画像_テスト.WEBP").write_bytes(_webp_bytes())

            self.app.input_path = str(input_dir)
            self.app.output_path = str(output_dir)
            self.app.preserve_structure = True
            processed = Image.new("RGB", (8, 8), color="blue")
            with mock.patch.object(main, "process_image_stream", return_value=processed):
                self.app.process_folder()

            expected_root = Path(
                main._structured_folder_output_dir(
                    str(input_dir),
                    str(output_dir),
                    self.app.mode,
                )
            )
            output = expected_root / "第１章_かなカナ" / "画像_テスト.WEBP.processed.png"
            self.assertEqual(self.app.total_files, 1)
            self.assertEqual(self.app.count_images_recursive(str(input_dir)), 1)
            self.assertTrue(output.exists())

    def test_archive_scans_and_processes_webp_with_japanese_paths(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive_path = root / "入力_日本語.zip"
            output_dir = root / "出力_日本語"
            output_dir.mkdir()
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("第１章_かなカナ/画像_テスト.webp", _webp_bytes())

            self.app.input_path = str(archive_path)
            self.app.output_path = str(output_dir)
            processed = Image.new("RGB", (8, 8), color="blue")
            with mock.patch.object(main, "process_image_stream", return_value=processed):
                self.app.process_archive()

            expected_root = Path(
                main._archive_output_dir(
                    str(archive_path),
                    str(output_dir),
                    self.app.mode,
                )
            )
            output = expected_root / "第１章_かなカナ" / "画像_テスト.webp.processed.png"
            self.assertEqual(self.app.total_files, 1)
            self.assertTrue(output.exists())


class JapaneseFontTests(unittest.TestCase):
    def test_official_noto_font_and_license_are_present(self):
        self.assertEqual(
            hashlib.sha256(NOTO_FONT.read_bytes()).hexdigest(),
            NOTO_FONT_SHA256,
        )
        self.assertIn(
            "SIL OPEN FONT LICENSE Version 1.1",
            NOTO_LICENSE.read_text(encoding="utf-8"),
        )

    def test_noto_font_renders_required_japanese_and_chinese_characters(self):
        font = ImageFont.truetype(NOTO_FONT, 18)
        missing = font.getmask(chr(0x10FFFF))
        missing_signature = (missing.size, bytes(missing))

        for character in "输入路径日本語あア々ー":
            with self.subTest(character=character):
                mask = font.getmask(character)
                self.assertNotEqual((mask.size, bytes(mask)), missing_signature)

    def test_path_controls_use_the_extended_font(self):
        app = main.DeepCreampyApp.__new__(main.DeepCreampyApp)
        app.path_font = None
        dpg.create_context()
        try:
            app.setup_font()
            with dpg.window():
                dpg.add_input_text(tag="输入路径")
                dpg.add_input_text(tag="输出路径")
                dpg.add_input_text(tag="日志输出")

            app.bind_path_font()

            self.assertIsNotNone(app.path_font)
            for item in ("输入路径", "输出路径", "日志输出"):
                self.assertEqual(dpg.get_item_font(item), app.path_font)
        finally:
            dpg.destroy_context()

    def test_pyinstaller_bundles_the_font_and_its_license(self):
        spec = (ROOT / "main.spec").read_text(encoding="utf-8")

        self.assertIn("font/NotoSansCJKjp-Regular.otf", spec)
        self.assertIn("font/NotoSansCJK-LICENSE.txt", spec)


if __name__ == "__main__":
    unittest.main()
