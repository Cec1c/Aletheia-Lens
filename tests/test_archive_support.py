import base64
import inspect
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile

import py7zr
from PIL import Image

import main


ROOT = Path(__file__).resolve().parents[1]
RAR5_SUBDIRS = base64.b64decode(
    "UmFyIRoHAQDz4YLrCwEFBwAGAQGAgIAAWyrxsjACAwuGAASGAKSDAsekBMmAAAESc3ViL2RpcjIvZmlsZTIudHh0CgMTCNwVX4XkBhNmaWxlMgokNHkgOAIDC4gABIgApIMCfSS3cYAAARpzdWIvd2l0aCBzcGFjZS9sb25nIGZuLnR4dAoDEyncFV9mv8sdbG9uZyBmbgoOjzxzOAIDC4UABIUApIMCwYnsL4AAARpzdWIvw7zItcSpw7bhuIvDqC9maWxlLnR4dAoDE0TdFV+dEHMIZmlsZQqvrxG4MAIDC4YABIYApIMCBPcp4oAAARJzdWIvZGlyMS9maWxlMS50eHQKAxP92xVfHJEnNGZpbGUxCtVl6Z4kAgMLAAUA7YMBAAAAAIAAAQhzdWIvZGlyMgoDEwjcFV/ICfsT1nxQqSoCAwsABQDtgwEAAAAAgAABDnN1Yi93aXRoIHNwYWNlCgMTKdwVX1vbgh6UOOweJQIDCwAFAO2DAQAAAACAAAEJc3ViL2VtcHR5CgMT5dsVX/bv4ArIG6fPLQIDCwAFAO2DAQAAAACAAAERc3ViL8O8yLXEqcO24biLw6gKAxNE3RVfmSwqCYEdQEkkAgMLAAUA7YMBAAAAAIAAAQhzdWIvZGlyMQoDE/3bFV8Nrd40msCgER8CAwsABQDtgwEAAAAAgAABA3N1YgoDEzLdFV+8OWMOHXdWUQMFBAA="
)


class ArchiveSupportTests(unittest.TestCase):
    def setUp(self):
        self.app = main.DeepCreampyApp.__new__(main.DeepCreampyApp)

    def test_extracts_nested_zip_content(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive_path = root / "images.zip"
            extraction_path = root / "extracted"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("chapter/page.png", b"png-data")

            with mock.patch.object(main.tempfile, "mkdtemp", return_value=str(extraction_path)):
                result = Path(self.app.extract_archive(str(archive_path)))

            self.assertEqual(result, extraction_path)
            self.assertEqual((result / "chapter" / "page.png").read_bytes(), b"png-data")

    def test_extracts_nested_7z_content(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "page.png"
            source.write_bytes(b"png-data")
            archive_path = root / "images.7z"
            extraction_path = root / "extracted"
            with py7zr.SevenZipFile(archive_path, "w") as archive:
                archive.write(source, arcname="chapter/page.png")

            with mock.patch.object(main.tempfile, "mkdtemp", return_value=str(extraction_path)):
                result = Path(self.app.extract_archive(str(archive_path)))

            self.assertEqual((result / "chapter" / "page.png").read_bytes(), b"png-data")

    def test_7z_rejects_link_metadata_before_extraction(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            target = root / "target.png"
            target.write_bytes(b"target")
            link = root / "link.png"
            try:
                os.symlink(target.name, link)
            except OSError as exc:
                self.skipTest(f"symbolic links are unavailable: {exc}")

            archive_path = root / "links.7z"
            with py7zr.SevenZipFile(archive_path, "w") as archive:
                archive.write(target, arcname="chapter/target.png")
                archive.write(link, arcname="chapter/link.png")

            with py7zr.SevenZipFile(archive_path, "r") as archive:
                self.assertTrue(any(member.is_symlink for member in archive.files))

            extraction_path = root / "extracted"
            with (
                mock.patch.object(main.tempfile, "mkdtemp", return_value=str(extraction_path)),
                mock.patch.object(
                    py7zr.SevenZipFile,
                    "extractall",
                    side_effect=AssertionError("unsafe extraction started"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "链接"):
                    self.app.extract_archive(str(archive_path))

            self.assertFalse(extraction_path.exists())

    def test_zip_rejects_unsafe_member_names_before_extraction(self):
        unsafe_names = (
            "../outside.png",
            "/absolute.png",
            "C:/drive.png",
            "chapter/file.png:stream",
            "chapter/a?.png",
            "chapter/a*.png",
            "chapter/a<.png",
            "chapter/a>.png",
            'chapter/a".png',
            "chapter/a|.png",
            "chapter/control\x01.png",
            "chapter/CON.png",
            "chapter/trailing-dot./page.png",
        )

        for member_name in unsafe_names:
            with self.subTest(member_name=member_name), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                archive_path = root / "unsafe.zip"
                extraction_path = root / "extracted"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr(member_name, b"unsafe")

                with (
                    mock.patch.object(main.tempfile, "mkdtemp", return_value=str(extraction_path)),
                    mock.patch.object(
                        zipfile.ZipFile,
                        "extractall",
                        side_effect=AssertionError("unsafe extraction started"),
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "路径|文件名"):
                        self.app.extract_archive(str(archive_path))

                self.assertFalse(extraction_path.exists())

    def test_zip_rejects_casefolded_output_collisions_before_extraction(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive_path = root / "collision.zip"
            extraction_path = root / "extracted"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("chapter/Page.png", b"first")
                archive.writestr("chapter/page.png", b"second")

            with (
                mock.patch.object(main.tempfile, "mkdtemp", return_value=str(extraction_path)),
                mock.patch.object(
                    zipfile.ZipFile,
                    "extractall",
                    side_effect=AssertionError("unsafe extraction started"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "冲突"):
                    self.app.extract_archive(str(archive_path))

            self.assertFalse(extraction_path.exists())

    def test_extracts_real_rar5_subdirectories_and_unicode_names(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive_path = root / "images.rar"
            archive_path.write_bytes(RAR5_SUBDIRS)
            extraction_path = root / "extracted"

            with mock.patch.object(main.tempfile, "mkdtemp", return_value=str(extraction_path)):
                result = Path(self.app.extract_archive(str(archive_path)))

            self.assertEqual((result / "sub" / "dir1" / "file1.txt").read_bytes(), b"file1\n")
            self.assertEqual((result / "sub" / "with space" / "long fn.txt").read_bytes(), b"long fn\n")
            self.assertEqual((result / "sub" / "üȵĩöḋè" / "file.txt").read_bytes(), b"file\n")

    def test_unsupported_archive_cleans_temporary_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive_path = root / "images.tar"
            archive_path.write_bytes(b"not-an-archive")
            extraction_path = root / "extracted"

            with mock.patch.object(main.tempfile, "mkdtemp", return_value=str(extraction_path)):
                with self.assertRaisesRegex(ValueError, "不支持的压缩包格式"):
                    self.app.extract_archive(str(archive_path))

            self.assertFalse(extraction_path.exists())

    def test_archive_rejects_oversized_contents_before_extraction(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive_path = root / "images.zip"
            extraction_path = root / "extracted"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("chapter/page.png", b"too-large")

            with (
                mock.patch.object(main.tempfile, "mkdtemp", return_value=str(extraction_path)),
                mock.patch.object(
                    main,
                    "MAX_ARCHIVE_UNCOMPRESSED_BYTES",
                    3,
                    create=True,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "解压后总大小"):
                    self.app.extract_archive(str(archive_path))

            self.assertFalse(extraction_path.exists())

    def test_archive_rejects_symbolic_links_and_cleans_temporary_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive_path = root / "images.zip"
            extraction_path = root / "extracted"
            link_info = zipfile.ZipInfo("chapter/link.png")
            link_info.create_system = 3
            link_info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(link_info, "../outside.png")

            with mock.patch.object(main.tempfile, "mkdtemp", return_value=str(extraction_path)):
                with self.assertRaisesRegex(ValueError, "符号链接"):
                    self.app.extract_archive(str(archive_path))

            self.assertFalse(extraction_path.exists())

    def test_packaged_build_bundles_a_rar_extractor(self):
        spec = (ROOT / "main.spec").read_text(encoding="utf-8")
        source = (ROOT / "main.py").read_text(encoding="utf-8")

        self.assertIn("7z.exe", spec)
        self.assertIn("shutil.which", spec)
        self.assertIn("raise FileNotFoundError", spec)
        self.assertIn("SEVENZIP_TOOL", source)

    def test_folder_processing_rejects_output_inside_input_tree(self):
        with tempfile.TemporaryDirectory() as root:
            input_dir = Path(root) / "input"
            input_dir.mkdir()
            self.app.input_path = str(input_dir)
            self.app.output_path = str(input_dir / "output")
            self.app.preserve_structure = False
            self.app.log_message = mock.Mock()

            with mock.patch.object(self.app, "get_all_image_files", return_value=[]) as scan:
                self.app.process_folder()

            scan.assert_not_called()
            self.assertIn(
                "输出目录不能位于输入目录内",
                self.app.log_message.call_args.args[0],
            )

    def test_processed_archive_images_use_png_extension_and_encoding(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            extracted = root / "extracted"
            chapter = extracted / "chapter"
            chapter.mkdir(parents=True)
            Image.new("RGB", (4, 4), color="red").save(chapter / "page.jpg", format="JPEG")

            output = root / "output"
            self.app.input_path = str(root / "images.zip")
            self.app.output_path = str(output)
            self.app.processing = True
            self.app.mode = 1
            self.app.total_files = 1
            self.app.current_file_index = 0
            self.app.log_message = mock.Mock()
            self.app.update_progress = mock.Mock()

            processed = Image.new("RGB", (4, 4), color="blue")
            with (
                mock.patch.object(self.app, "extract_archive", return_value=str(extracted)),
                mock.patch.object(main, "process_image_stream", return_value=processed),
            ):
                self.app.process_archive()

            archive_output = Path(
                main._archive_output_dir(
                    str(root / "images.zip"),
                    str(output),
                    self.app.mode,
                )
            )
            output_path = archive_output / "chapter" / "page.jpg.processed.png"
            self.assertTrue(output_path.exists())
            self.assertFalse((archive_output / "chapter" / "page.jpg").exists())
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_processed_folder_images_use_png_extension(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            input_dir = root / "input"
            input_dir.mkdir()
            source = input_dir / "page.jpg"
            Image.new("RGB", (4, 4), color="red").save(source, format="JPEG")

            self.app.processing = True
            self.app.mode = 1
            self.app.total_files = 1
            self.app.current_file_index = 0
            self.app.log_message = mock.Mock()
            self.app.update_progress = mock.Mock()
            processed = Image.new("RGB", (4, 4), color="blue")

            with mock.patch.object(main, "process_image_stream", return_value=processed):
                self.app.process_with_structure(
                    str(input_dir),
                    str(root / "output"),
                    [(str(input_dir), "page.jpg", ".")],
                )

            output_path = (
                Path(
                    main._structured_folder_output_dir(
                        str(input_dir),
                        str(root / "output"),
                        self.app.mode,
                    )
                )
                / "page.jpg.processed.png"
            )
            self.assertTrue(output_path.exists())
            self.assertFalse(output_path.with_name("page.jpg").exists())

    def test_structured_output_namespaces_include_source_identity_and_mode(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first = root / "first" / "images"
            second = root / "second" / "images"
            output = root / "output"

            namespaces = {
                main._structured_folder_output_dir(str(first), str(output), 1),
                main._structured_folder_output_dir(str(second), str(output), 1),
                main._structured_folder_output_dir(str(first), str(output), 2),
            }

        self.assertEqual(len(namespaces), 3)
        self.assertTrue(all(Path(path).name.startswith("after_images_") for path in namespaces))

    def test_archive_output_namespaces_include_source_identity_and_mode(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first = root / "first" / "chapter.zip"
            second = root / "second" / "chapter.zip"
            output = root / "output"

            namespaces = {
                main._archive_output_dir(str(first), str(output), 1),
                main._archive_output_dir(str(second), str(output), 1),
                main._archive_output_dir(str(first), str(output), 2),
            }

        self.assertEqual(len(namespaces), 3)
        self.assertTrue(all(Path(path).name.startswith("after_chapter_") for path in namespaces))

    def test_processed_flat_folder_images_use_png_extension(self):
        self.assertTrue(hasattr(main, "_flat_png_output_name"))
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            input_dir = root / "input"
            input_dir.mkdir()
            source = input_dir / "page.jpg"
            Image.new("RGB", (4, 4), color="red").save(source, format="JPEG")

            self.app.processing = True
            self.app.mode = 1
            self.app.total_files = 1
            self.app.current_file_index = 0
            self.app.log_message = mock.Mock()
            self.app.update_progress = mock.Mock()
            processed = Image.new("RGB", (4, 4), color="blue")
            output = root / "output"

            with mock.patch.object(main, "process_image_stream", return_value=processed):
                self.app.process_flat(
                    str(input_dir),
                    str(output),
                    [(str(input_dir), "page.jpg", ".")],
                )

            output_name = main._flat_png_output_name(
                str(input_dir),
                ".",
                "page.jpg",
                self.app.mode,
            )
            self.assertTrue((output / output_name).exists())
            self.assertFalse((output / "page.jpg").exists())

    def test_flat_output_names_do_not_collide_when_paths_flatten_similarly(self):
        self.assertTrue(hasattr(main, "_flat_png_output_name"))
        sources = [
            (".", "a_b_page.jpg"),
            (os.path.join("a", "b"), "page.jpg"),
            ("a_b", "page.jpg"),
        ]
        output_names = [
            main._flat_png_output_name("input-root", path, filename, 1)
            for path, filename in sources
        ]

        self.assertEqual(len(set(output_names)), len(sources))
        self.assertTrue(all(name.endswith(".processed.png") for name in output_names))

    def test_flat_output_names_include_input_root_and_processing_mode(self):
        self.assertEqual(
            list(inspect.signature(main._flat_png_output_name).parameters),
            ["input_dir", "relative_path", "filename", "mode"],
        )

        with tempfile.TemporaryDirectory() as root:
            first_root = Path(root) / "first"
            second_root = Path(root) / "second"
            sources = [
                (str(first_root), "chapter", "page.jpg", 1),
                (str(second_root), "chapter", "page.jpg", 1),
                (str(first_root), "chapter", "page.jpg", 2),
            ]
            output_names = [main._flat_png_output_name(*source) for source in sources]

        self.assertEqual(len(set(output_names)), len(sources))

    def test_png_output_names_do_not_collide_across_source_extensions(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            input_dir = root / "input"
            input_dir.mkdir()
            Image.new("RGB", (4, 4), color="red").save(input_dir / "page.jpg", format="JPEG")
            Image.new("RGB", (4, 4), color="green").save(input_dir / "page.jpg.png", format="PNG")

            self.app.processing = True
            self.app.mode = 1
            self.app.total_files = 2
            self.app.current_file_index = 0
            self.app.log_message = mock.Mock()
            self.app.update_progress = mock.Mock()

            with mock.patch.object(
                main,
                "process_image_stream",
                side_effect=[
                    Image.new("RGB", (4, 4), color="blue"),
                    Image.new("RGB", (4, 4), color="yellow"),
                ],
            ):
                self.app.process_with_structure(
                    str(input_dir),
                    str(root / "output"),
                    [
                        (str(input_dir), "page.jpg", "."),
                        (str(input_dir), "page.jpg.png", "."),
                    ],
                )

            output_dir = Path(
                main._structured_folder_output_dir(
                    str(input_dir),
                    str(root / "output"),
                    self.app.mode,
                )
            )
            self.assertTrue((output_dir / "page.jpg.processed.png").exists())
            self.assertTrue((output_dir / "page.jpg.png.processed.png").exists())


if __name__ == "__main__":
    unittest.main()
