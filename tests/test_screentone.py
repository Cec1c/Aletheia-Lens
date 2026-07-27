import io
import sys
from types import ModuleType
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import main
from screentone import remove_screentones


def _png_bytes(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _high_frequency_energy(image):
    pixels = np.asarray(image, dtype=np.float32)
    vertical = np.abs(np.diff(pixels, axis=0)).mean()
    horizontal = np.abs(np.diff(pixels, axis=1)).mean()
    return float(vertical + horizontal)


def _load_processer_without_models():
    detector_module = ModuleType("detector")
    detector_module.detector = mock.Mock()
    detector_module.apply_cover = mock.Mock()
    decensor_module = ModuleType("decensor")
    decensor_module.decensor = mock.Mock()
    esrgan_module = ModuleType("esrgan")
    esrgan_module.esrgan = mock.Mock()

    with mock.patch.dict(
        sys.modules,
        {
            "detector": detector_module,
            "decensor": decensor_module,
            "esrgan": esrgan_module,
        },
    ):
        sys.modules.pop("processer", None)
        import processer

    sys.modules.pop("processer", None)
    return processer


class ScreentoneFilterTests(unittest.TestCase):
    def test_invalid_strength_is_rejected(self):
        image = Image.new("RGB", (8, 8), color="white")

        for level in (0, 4, -1, None):
            with self.subTest(level=level):
                with self.assertRaisesRegex(ValueError, "1、2 或 3"):
                    remove_screentones(image, level)

    def test_dimensions_are_unchanged(self):
        image = Image.new("RGB", (37, 23), color=(12, 34, 56))

        for level in (1, 2, 3):
            with self.subTest(level=level):
                self.assertEqual(remove_screentones(image, level).size, image.size)

    def test_constant_rgb_image_does_not_drift(self):
        pixels = np.full((25, 27, 3), (43, 87, 129), dtype=np.uint8)
        image = Image.fromarray(pixels)

        for level in (1, 2, 3):
            with self.subTest(level=level):
                filtered = np.asarray(remove_screentones(image, level))
                np.testing.assert_array_equal(filtered, pixels)

    def test_rgba_alpha_bytes_are_preserved_exactly(self):
        y, x = np.indices((31, 29))
        rgba = np.empty((31, 29, 4), dtype=np.uint8)
        rgba[:, :, 0] = (x * 17 + y * 3) % 256
        rgba[:, :, 1] = (x * 5 + y * 11) % 256
        rgba[:, :, 2] = (x * 7 + y * 13) % 256
        rgba[:, :, 3] = (x * 19 + y * 23) % 256
        image = Image.fromarray(rgba)

        for level in (1, 2, 3):
            with self.subTest(level=level):
                filtered = remove_screentones(image, level)
                self.assertEqual(filtered.mode, "RGBA")
                np.testing.assert_array_equal(np.asarray(filtered)[:, :, 3], rgba[:, :, 3])

    def test_high_frequency_screentone_energy_is_reduced(self):
        y, x = np.indices((128, 128))
        dots = np.where(((x // 2 + y // 2) % 2) == 0, 60, 220).astype(np.uint8)
        image = Image.fromarray(np.repeat(dots[:, :, None], 3, axis=2))
        original_energy = _high_frequency_energy(image)

        filtered_energy = _high_frequency_energy(remove_screentones(image, 1))

        self.assertLess(filtered_energy, original_energy * 0.25)

    def test_strong_filter_is_not_weaker_than_light_filter(self):
        y, x = np.indices((128, 128))
        dots = np.where(((x // 2 + y // 2) % 2) == 0, 60, 220).astype(np.uint8)
        image = Image.fromarray(np.repeat(dots[:, :, None], 3, axis=2))

        light_energy = _high_frequency_energy(remove_screentones(image, 1))
        strong_energy = _high_frequency_energy(remove_screentones(image, 3))

        self.assertLessEqual(strong_energy, light_energy)

    def test_major_edge_is_retained(self):
        pixels = np.zeros((128, 128, 3), dtype=np.uint8)
        pixels[:, 64:] = 255
        image = Image.fromarray(pixels)

        for level in (1, 2, 3):
            with self.subTest(level=level):
                filtered = np.asarray(remove_screentones(image, level), dtype=np.float32)
                left_mean = filtered[:, :48].mean()
                right_mean = filtered[:, 80:].mean()
                self.assertGreater(right_mean - left_mean, 220)


class ScreentoneIntegrationTests(unittest.TestCase):
    def test_process_image_stream_only_preprocesses_when_enabled(self):
        processer = _load_processer_without_models()
        source = _png_bytes(Image.new("RGB", (4, 4), color="white"))
        expected = Image.new("RGB", (4, 4), color="blue")

        with (
            mock.patch.object(
                processer,
                "apply_screentone_preprocessing",
                return_value=b"filtered",
            ) as preprocess,
            mock.patch.object(
                processer,
                "process_bar_auto",
                return_value=expected,
            ) as repair,
        ):
            self.assertIs(processer.process_image_stream(source, 1), expected)
            preprocess.assert_not_called()
            repair.assert_called_once_with(source)

            repair.reset_mock()
            self.assertIs(
                processer.process_image_stream(source, 1, screentone_level=2),
                expected,
            )
            preprocess.assert_called_once_with(source, 2)
            repair.assert_called_once_with(b"filtered")

    def test_gui_callbacks_and_processor_parameter_forwarding(self):
        app = main.DeepCreampyApp.__new__(main.DeepCreampyApp)
        app.mode = 2
        app.screentone_enabled = False
        app.screentone_level = 2

        with mock.patch.object(main.dpg, "configure_item") as configure_item:
            app.on_screentone_enabled_change(None, True)
        configure_item.assert_called_once_with("去网点强度", enabled=True)

        app.on_screentone_level_change(None, "强度")
        self.assertEqual(app.active_screentone_level(), 3)

        result = object()
        with mock.patch.object(main, "process_image_stream", return_value=result) as process:
            self.assertIs(app.process_image_bytes(b"source"), result)
        process.assert_called_once_with(b"source", 2, screentone_level=3)

        with mock.patch.object(main.dpg, "configure_item"):
            app.on_screentone_enabled_change(None, False)
        self.assertEqual(app.active_screentone_level(), 0)

    def test_output_namespaces_include_screentone_strength(self):
        levels = (0, 1, 2, 3)
        flat_names = {
            main._flat_png_output_name("input", "chapter", "page.png", 1, level)
            for level in levels
        }
        folder_names = {
            main._structured_folder_output_dir("input", "output", 1, level)
            for level in levels
        }
        archive_names = {
            main._archive_output_dir("input.zip", "output", 1, level)
            for level in levels
        }

        self.assertEqual(len(flat_names), len(levels))
        self.assertEqual(len(folder_names), len(levels))
        self.assertEqual(len(archive_names), len(levels))


if __name__ == "__main__":
    unittest.main()
