import unittest
from unittest import mock
from pathlib import Path

import numpy as np

import detector
import esrgan
import config


class _NamedOutputSession:
    def __init__(self):
        self.requested_outputs = None

    def run(self, output_names, inputs):
        self.requested_outputs = output_names
        if output_names != ["mrcnn_detection", "mrcnn_mask"]:
            raise AssertionError(
                "Mask R-CNN outputs must be requested by name; ONNX output order is not stable"
            )

        detections = np.zeros((1, 100, 6), dtype=np.float32)
        masks = np.zeros((1, 100, 28, 28, 3), dtype=np.float32)
        return detections, masks


class _EchoEsrganSession:
    def run(self, output_names, inputs):
        self.output_names = output_names
        self.input_tensor = inputs["input"]
        return [np.repeat(np.repeat(self.input_tensor, 4, axis=2), 4, axis=3)]


class OnnxMigrationTests(unittest.TestCase):
    @unittest.skipUnless(Path(config.mrcnn_model).exists(), "Mask R-CNN ONNX model is not installed")
    def test_real_mask_rcnn_model_handles_an_image_with_no_detections(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        result = detector.detect_image(image)

        self.assertEqual(result["scores"].shape, (0,))
        self.assertEqual(result["masks"].shape, (64, 64, 0))

    @unittest.skipUnless(Path(config.mrcnn_model).exists(), "Mask R-CNN ONNX model is not installed")
    def test_real_mask_rcnn_model_detects_a_synthetic_mosaic(self):
        image = np.full((512, 512, 3), 255, dtype=np.uint8)
        for y in range(160, 352, 16):
            for x in range(160, 352, 16):
                image[y : y + 16, x : x + 16] = 0 if ((x + y) // 16) % 2 == 0 else 128

        result = detector.detect_image(image)
        mosaic_indices = np.where(result["class_ids"] == 2)[0]

        self.assertGreater(len(mosaic_indices), 0)
        best = mosaic_indices[np.argmax(result["scores"][mosaic_indices])]
        self.assertGreater(float(result["scores"][best]), 0.9)
        actual_mask = result["masks"][:, :, best].astype(bool)
        expected_mask = np.zeros((512, 512), dtype=bool)
        expected_mask[160:352, 160:352] = True
        mask_intersection = np.logical_and(actual_mask, expected_mask).sum()
        mask_union = np.logical_or(actual_mask, expected_mask).sum()
        self.assertGreater(mask_intersection / float(mask_union), 0.75)

        expected = np.array([160, 160, 352, 352], dtype=np.int32)
        actual = result["rois"][best]
        intersection_y1 = max(expected[0], actual[0])
        intersection_x1 = max(expected[1], actual[1])
        intersection_y2 = min(expected[2], actual[2])
        intersection_x2 = min(expected[3], actual[3])
        intersection = max(0, intersection_y2 - intersection_y1) * max(
            0, intersection_x2 - intersection_x1
        )
        expected_area = (expected[2] - expected[0]) * (expected[3] - expected[1])
        actual_area = (actual[2] - actual[0]) * (actual[3] - actual[1])
        iou = intersection / float(expected_area + actual_area - intersection)
        self.assertGreater(iou, 0.7)

    def test_mask_rcnn_requests_detection_and_mask_outputs_by_name(self):
        fake_session = _NamedOutputSession()
        image = np.zeros((32, 48, 3), dtype=np.uint8)

        with mock.patch.object(detector, "_onnx_session", fake_session):
            result = detector._detect_onnx(image)

        self.assertEqual(
            fake_session.requested_outputs,
            ["mrcnn_detection", "mrcnn_mask"],
        )
        self.assertEqual(result["masks"].shape, (32, 48, 0))

    def test_denormalized_boxes_are_integer_pixel_coordinates(self):
        boxes = detector._denorm_boxes(
            np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
            (10, 20),
        )

        self.assertTrue(np.issubdtype(boxes.dtype, np.integer))
        np.testing.assert_array_equal(boxes, [[0, 0, 10, 20]])

    def test_mode_three_uses_the_onnx_detector_interface(self):
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        detection = {
            "rois": np.array([[0, 0, 4, 4], [1, 1, 5, 5]], dtype=np.int32),
            "class_ids": np.array([2, 1], dtype=np.int32),
            "scores": np.array([0.9, 0.8], dtype=np.float32),
            "masks": np.zeros((8, 10, 2), dtype=np.bool_),
        }
        detection["masks"][1:4, 2:5, 0] = True

        with mock.patch.object(
            esrgan.detector,
            "detect_image",
            return_value=detection,
            create=True,
        ) as detect_image:
            masks = esrgan.get_masks(image)

        self.assertEqual(detect_image.call_count, 1)
        np.testing.assert_array_equal(detect_image.call_args.args[0], image)
        self.assertEqual(masks.shape, (8, 10, 1))
        self.assertTrue(masks[1:4, 2:5, 0].all())

    def test_esrgan_preserves_internal_rgb_channel_order(self):
        fake_session = _EchoEsrganSession()
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)

        with mock.patch.object(esrgan, "_esrgan_session", fake_session):
            result = esrgan._run_esrgan_onnx(image)

        np.testing.assert_allclose(
            fake_session.input_tensor[0, :, 0, 0],
            np.array([10, 20, 30], dtype=np.float32) / 255.0,
        )
        np.testing.assert_array_equal(result[0, 0], image[0, 0])

    def test_conversion_scripts_fail_closed_on_invalid_model_results(self):
        esrgan_source = (Path(__file__).resolve().parents[1] / "convert_esrgan_to_onnx.py").read_text(
            encoding="utf-8"
        )
        mrcnn_source = (Path(__file__).resolve().parents[1] / "convert_mrcnn_to_onnx.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("np.isfinite", esrgan_source)
        self.assertIn("raise RuntimeError", esrgan_source)
        self.assertNotIn("精度有偏差", esrgan_source)
        self.assertIn("np.full((512, 512, 3)", mrcnn_source)
        self.assertIn("result['class_ids'] == 2", mrcnn_source)
        self.assertIn("mask_iou", mrcnn_source)
        self.assertNotIn("无检测烟测", mrcnn_source)


if __name__ == "__main__":
    unittest.main()
