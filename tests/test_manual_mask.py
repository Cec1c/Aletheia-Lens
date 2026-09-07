import io
from pathlib import Path
import queue
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import decensor
import detector
import main
from mask_editor import CanvasTransform, MaskDocument, MaskEditor, render_canvas, save_manual_result
from processer import prepare_manual_image, repair_manual_bars


class MaskDocumentTests(unittest.TestCase):
    def test_stroke_interpolates_and_erase_undo_redo_preserve_source(self):
        image = Image.new("RGB", (120, 80), "green")
        doc = MaskDocument(image)
        doc.begin_stroke()
        doc.paint((10, 25), 5)
        doc.paint((100, 25), 5)
        self.assertTrue(doc.end_stroke())
        painted = doc.mask.copy()
        self.assertTrue(doc.mask[25, 10:101].all())
        doc.begin_stroke()
        doc.paint((55, 25), 12, erase=True)
        doc.end_stroke()
        self.assertFalse(doc.mask[25, 55])
        self.assertTrue(doc.undo())
        np.testing.assert_array_equal(doc.mask, painted)
        self.assertTrue(doc.redo())
        self.assertFalse(doc.mask[25, 55])
        self.assertEqual(image.tobytes(), doc.image.tobytes())

    def test_leaving_image_breaks_stroke_and_edges_are_clipped(self):
        doc = MaskDocument(Image.new("RGB", (100, 60)))
        doc.begin_stroke()
        doc.paint((0, 0), 10)
        doc.paint(None, 10)
        doc.paint((99, 59), 10)
        doc.end_stroke()
        self.assertTrue(doc.mask[0, 0])
        self.assertTrue(doc.mask[59, 99])
        self.assertFalse(doc.mask[30, 50])

    def test_expansion_is_preview_only_until_applied_and_can_be_undone(self):
        doc = MaskDocument(Image.new("RGB", (30, 30)))
        doc.mask[15, 15] = True
        preview = doc.expanded(3)
        self.assertEqual(doc.mask.sum(), 1)
        self.assertGreater(preview.sum(), 1)
        doc.replace(preview)
        doc.undo()
        self.assertEqual(doc.mask.sum(), 1)

    def test_history_is_bounded_and_new_edit_clears_redo(self):
        doc = MaskDocument(Image.new("RGB", (8, 8)), history_bytes=16)
        for x in range(5):
            mask = doc.mask.copy()
            mask[0, x] = True
            doc.replace(mask)
        self.assertEqual(len(doc.undo_stack), 2)
        doc.undo()
        doc.replace(np.zeros_like(doc.mask))
        self.assertFalse(doc.redo())

    def test_invalid_replacement_is_rejected(self):
        doc = MaskDocument(Image.new("RGB", (20, 10)))
        for mask in (np.zeros((20, 10), bool), np.zeros((10, 20), np.uint8)):
            with self.assertRaises(ValueError):
                doc.replace(mask)


class CanvasTests(unittest.TestCase):
    def test_letterboxing_zoom_anchor_and_pan_map_to_original_pixels(self):
        transform = CanvasTransform(1000, 500, 500, 500)
        transform.fit()
        self.assertEqual(transform.to_image((250, 250)), (500, 250))
        self.assertIsNone(transform.to_image((250, 100)))
        self.assertIsNone(transform.to_image((500, 250)))
        transform.zoom(2, (250, 250))
        self.assertEqual(transform.to_image((250, 250)), (500, 250))
        transform.offset_x += 70
        transform.offset_y -= 40
        self.assertEqual(transform.to_image((320, 210)), (500, 250))

    def test_render_overlay_matches_mapping_without_changing_mask(self):
        image = Image.new("RGBA", (100, 50), (180, 40, 30, 255))
        mask = np.zeros((50, 100), dtype=bool)
        mask[20:30, 45:55] = True
        transform = CanvasTransform(100, 50, 200, 200)
        transform.fit()
        canvas = render_canvas(image, mask, transform).reshape(200, 200, 4)
        self.assertGreater(canvas[100, 100, 1], canvas[100, 20, 1])
        self.assertEqual(mask.sum(), 100)
        np.testing.assert_allclose(canvas[:, :, 3], 1)

    def test_cached_premultiplied_preview_matches_transparent_rgba_render(self):
        pixels = np.zeros((50, 100, 4), dtype=np.uint8)
        pixels[:, :, :3] = (190, 80, 50)
        pixels[:, :, 3] = np.arange(100) * 2
        image = Image.fromarray(pixels)
        transform = CanvasTransform(100, 50, 200, 200)
        transform.fit()
        np.testing.assert_array_equal(render_canvas(image, None, transform),
                                      render_canvas(image.convert("RGBa"), None, transform))


class ManualPipelineTests(unittest.TestCase):
    def test_detection_selects_only_bars_and_zero_detection_is_empty(self):
        image = Image.new("RGB", (9, 7), (0, 255, 0))
        masks = np.zeros((7, 9, 2), dtype=bool)
        masks[1, 1, 0] = True
        masks[4, 5, 1] = True
        with mock.patch.object(detector, "detect_image", return_value={"masks": masks, "class_ids": np.array([1, 2])}):
            actual = detector.detect_bar_mask(image)
        self.assertEqual(actual.sum(), 1)
        self.assertTrue(actual[1, 1])
        with mock.patch.object(detector, "detect_image", return_value={"masks": masks[:, :, :0], "class_ids": np.array([], dtype=int)}):
            self.assertFalse(detector.detect_bar_mask(image).any())

    def test_explicit_mask_preserves_green_unselected_pixels_and_alpha(self):
        y, x = np.indices((300, 320))
        pixels = np.empty((300, 320, 4), dtype=np.uint8)
        pixels[:, :, :3] = (0, 255, 0)
        pixels[:, :, 3] = (x + y) % 256
        source = Image.fromarray(pixels)
        mask = np.zeros((300, 320), dtype=bool)
        mask[100:110, 140:155] = True
        with mock.patch.object(decensor, "predict", return_value=np.zeros((256, 256, 3), np.float32)) as predict:
            output = np.asarray(repair_manual_bars(source, mask))
        predict.assert_called_once()
        np.testing.assert_array_equal(output[~mask], pixels[~mask])
        np.testing.assert_array_equal(output[:, :, 3], pixels[:, :, 3])
        self.assertTrue((output[mask, :3] == 127).all())
        np.testing.assert_array_equal(np.asarray(source), pixels)

    def test_empty_mask_skips_inference_even_on_natural_green(self):
        source = Image.new("RGBA", (25, 20), (0, 255, 0, 121))
        with mock.patch.object(decensor, "predict") as predict:
            output = repair_manual_bars(source, np.zeros((20, 25), dtype=bool))
        predict.assert_not_called()
        self.assertEqual(source.tobytes(), output.tobytes())
        self.assertIsNot(source, output)

    def test_invalid_inference_output_is_not_silently_saved_as_black(self):
        source = Image.new("RGB", (300, 300), "white")
        mask = np.zeros((300, 300), dtype=bool)
        mask[140:150, 140:150] = True
        with mock.patch.object(decensor, "predict", return_value=np.full((256, 256, 3), np.nan)):
            with self.assertRaisesRegex(ValueError, "无效图像"):
                repair_manual_bars(source, mask)

    def test_thin_images_and_border_selection_repair_every_selected_pixel(self):
        for size in ((640, 80), (80, 640), (9, 7)):
            with self.subTest(size=size):
                source = Image.new("RGBA", size, (0, 0, 0, 77))
                mask = np.zeros((size[1], size[0]), dtype=bool)
                mask[0, :] = True
                mask[:, -1] = True
                with mock.patch.object(decensor, "predict", return_value=np.zeros((256, 256, 3), np.float32)):
                    result = np.asarray(repair_manual_bars(source, mask))
                self.assertTrue((result[mask, :3] == 127).all())
                self.assertTrue((result[~mask, :3] == 0).all())
                self.assertTrue((result[:, :, 3] == 77).all())

    def test_mask_shape_and_type_fail_before_inference(self):
        image = Image.new("RGB", (30, 20))
        with mock.patch.object(decensor, "predict") as predict:
            for mask in (np.zeros((20, 30), np.uint8), np.zeros((30, 20), bool)):
                with self.assertRaises(ValueError):
                    repair_manual_bars(image, mask)
        predict.assert_not_called()

    def test_manual_regions_keep_diagonal_marks_separate(self):
        mask = np.eye(3, dtype=bool)
        self.assertEqual([len(region) for region in decensor.find_mask_regions(mask)], [1, 1, 1])

    def test_preprocessing_runs_once_before_repeated_repairs(self):
        source = Image.new("RGBA", (40, 30), (12, 34, 56, 89))
        stream = io.BytesIO()
        source.save(stream, "PNG")
        with mock.patch("processer.remove_screentones", return_value=source.copy()) as preprocess:
            working = prepare_manual_image(stream.getvalue(), 2)
            for _ in range(2):
                repair_manual_bars(working, np.zeros((30, 40), bool))
        preprocess.assert_called_once()
        self.assertEqual(working.tobytes(), source.tobytes())

    def test_decoder_preserves_palette_transparency_and_applies_exif_orientation(self):
        palette = Image.new("P", (4, 3), 0)
        palette.info["transparency"] = 0
        stream = io.BytesIO()
        palette.save(stream, "PNG")
        prepared = prepare_manual_image(stream.getvalue())
        self.assertEqual(prepared.mode, "RGBA")
        self.assertFalse(np.asarray(prepared)[:, :, 3].any())
        rgb = Image.new("RGB", (40, 20))
        exif = Image.Exif()
        exif[274] = 6
        stream = io.BytesIO()
        rgb.save(stream, "JPEG", exif=exif)
        self.assertEqual(prepare_manual_image(stream.getvalue()).size, (20, 40))

    def test_animation_is_rejected(self):
        stream = io.BytesIO()
        Image.new("RGB", (5, 5), "red").save(stream, "PNG", save_all=True,
                                             append_images=[Image.new("RGB", (5, 5), "blue")])
        with self.assertRaisesRegex(ValueError, "静态"):
            prepare_manual_image(stream.getvalue())

    def test_unicode_save_is_non_overwriting_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "画像_日本語.png"
            source.write_bytes(b"original")
            image = Image.new("RGBA", (8, 9), (20, 30, 40, 50))
            first = save_manual_result(image, source, directory, 2)
            second = save_manual_result(image, source, directory, 2)
            self.assertNotEqual(first, second)
            self.assertIn(".descreen-2", first)
            self.assertEqual(source.read_bytes(), b"original")
            with Image.open(second) as result:
                self.assertEqual(result.tobytes(), image.tobytes())


class ManualSessionTests(unittest.TestCase):
    def test_callback_failure_does_not_stop_the_render_loop_or_next_callback(self):
        app = main.DeepCreampyApp.__new__(main.DeepCreampyApp)
        app.log_message = mock.Mock()
        completed = []

        def fail():
            raise ValueError("test callback failure")

        def succeed():
            completed.append(True)

        with mock.patch.object(main.dpg, "get_callback_queue", return_value=[
            [fail, None, None, None], [succeed, None, None, None],
        ]):
            app.dispatch_ui_callbacks()
        self.assertEqual(completed, [True])
        app.log_message.assert_called_once()

    def test_closed_editor_discards_late_completion_without_gui_access(self):
        editor = MaskEditor.__new__(MaskEditor)
        editor.completions = queue.Queue()
        editor.completions.put((1, "repair", object(), None))
        editor.closed = True
        editor.busy = True
        editor.generation = 2
        editor.result = None
        editor._controls = mock.Mock()
        editor._drain()
        self.assertFalse(editor.busy)
        self.assertIsNone(editor.result)
        editor._controls.assert_not_called()

    def test_worker_failure_keeps_editable_document(self):
        editor = MaskEditor.__new__(MaskEditor)
        editor.completions = queue.Queue()
        editor.completions.put((1, "repair", None, "test failure"))
        editor.closed = False
        editor.busy = True
        editor.generation = 1
        editor.document = MaskDocument(Image.new("RGB", (5, 5)))
        editor.document.mask[2, 2] = True
        editor._controls = mock.Mock()
        editor._status = mock.Mock()
        editor.log = mock.Mock()
        editor._drain()
        self.assertFalse(editor.busy)
        self.assertTrue(editor.document.mask[2, 2])
        editor._status.assert_called_once()

    def test_main_prevents_overlap_with_a_closed_but_running_editor(self):
        app = main.DeepCreampyApp.__new__(main.DeepCreampyApp)
        app.processing = False
        app.manual_editor = mock.Mock(closed=True, busy=True)
        app.log_message = mock.Mock()
        with mock.patch.object(main.dpg, "get_value") as read:
            app.execute_processing()
        read.assert_not_called()
        app.log_message.assert_called_once()


if __name__ == "__main__":
    unittest.main()
